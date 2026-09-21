from __future__ import annotations

import os
import json
import pickle
from datetime import datetime
from pathlib import Path

import psycopg2
from tqdm import tqdm


# ============================================================
# 0. DATASET DEFINITION / COHORT RULES
# ============================================================
#
# Study goal:
#   Longitudinal ECG + CXR -> TTE Echo-derived cardiac phenotypes
#
# One sample:
#   1 TTE Echo event (measurement_id) = 1 target sample
#
# Target:
#   mimiciv_echo.structured_measurement
#   - test_type = 'tte'
#   - keep all non-empty structured measurements for that Echo event
#
# Patient/master data layer:
#   echo_multimodal.echo_patients
#       -> all patients with Echo
#
#   echo_multimodal.ecg_master
#       -> all ECGs for Echo patients, WITHOUT cohort time-window restriction
#
#   echo_multimodal.cxr_master
#       -> all CXR images for Echo patients, WITHOUT cohort time-window restriction
#
# Cohort for this version (AP-only longitudinal CXR):
#   1) TTE Echo only
#   2) Historical window = [Echo - 180 days, Echo)
#   3) >= 2 DISTINCT ECG studies in the 180-day history
#   4) CXR view_position = 'AP' only
#   5) >= 2 DISTINCT AP CXR studies in the 180-day history
#   6) 1 CXR study = 1 temporal event
#      If one study contains multiple AP images, deterministically keep ONE image
#      (ordered by dicom_id; first one retained)
#   7) No 72-hour recent-anchor restriction
#   8) No repeated-Echo deduplication / no minimum target gap
#   9) No hadm_id / hospitalization restriction
#
# Expected cohort from the user's current database:
#   ~9,657 Echo samples
#   ~5,560 patients
#
# IMPORTANT for later modeling:
#   train/val/test MUST be split by subject_id, not by sample_id.
#
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mimic",
    "user": "admin",
    'password': '123456'
}

# Recommended: export PGPASSWORD='your_password'
if os.getenv("PGPASSWORD"):
    DB_CONFIG["password"] = os.getenv("PGPASSWORD")

# These roots should be the directories that contain the dataset-level `files/` folder.
ECG_ROOT = Path("dataset/mimic-ecg")
CXR_ROOT = Path("dataset/mimic-cxr")

OUTPUT_DIR = Path("./dataset/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PKL = OUTPUT_DIR / "samples.pkl"
OUTPUT_JSONL = OUTPUT_DIR / "samples.jsonl"
PREVIEW_JSON = OUTPUT_DIR / "samples_preview.json"
META_FILE = OUTPUT_DIR / "dataset_meta.json"

EXPECTED_SAMPLES = 9657
EXPECTED_PATIENTS = 5560


# ============================================================
# 2. HELPERS
# ============================================================

def fetch_all(conn, sql: str):
    """Execute SQL and return (columns, rows)."""
    with conn.cursor() as cur:
        cur.execute(sql)
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    return columns, rows


def resolve_dataset_path(root: Path, path_value: str | None) -> str | None:
    """
    master tables may store either relative or absolute paths.
    - absolute path -> keep as-is
    - relative path -> prefix with dataset root
    """
    if path_value is None:
        return None

    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str(root / p)


def json_default(obj):
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return str(obj)


# ============================================================
# 3. CONNECT DATABASE
# ============================================================

print("Connecting PostgreSQL...")
conn = psycopg2.connect(**DB_CONFIG)
print("Connected.")


# ============================================================
# 4. CREATE TEMP TARGET COHORT
# ============================================================
#
# NOTE:
#   ECG and CXR are read from echo_multimodal.* master tables.
#   We do NOT go back to mimiciv_ecg.record_list or mimiciv_cxr.metadata.
#
# ============================================================

print("\nCreating temporary AP-only target cohort...")

create_target_sql = r"""
DROP TABLE IF EXISTS tmp_echo_targets;

CREATE TEMP TABLE tmp_echo_targets AS

WITH echo AS (
    SELECT DISTINCT
        s.subject_id,
        s.measurement_id,
        s.measurement_datetime AS echo_time
    FROM mimiciv_echo.structured_measurement s
    JOIN echo_multimodal.echo_patients ep
        ON ep.subject_id = s.subject_id
    WHERE s.test_type = 'tte'
      AND s.subject_id IS NOT NULL
      AND s.measurement_id IS NOT NULL
      AND s.measurement_datetime IS NOT NULL
),

ecg_count AS (
    SELECT
        e.subject_id,
        e.measurement_id,
        COUNT(DISTINCT g.ecg_study_id) AS ecg_n
    FROM echo e
    JOIN echo_multimodal.ecg_master g
        ON g.subject_id = e.subject_id
       AND g.ecg_time >= e.echo_time - INTERVAL '180 days'
       AND g.ecg_time <  e.echo_time
    GROUP BY
        e.subject_id,
        e.measurement_id
),

ap_cxr_count AS (
    SELECT
        e.subject_id,
        e.measurement_id,
        COUNT(DISTINCT c.cxr_study_id) AS ap_cxr_n
    FROM echo e
    JOIN echo_multimodal.cxr_master c
        ON c.subject_id = e.subject_id
       AND c.cxr_time >= e.echo_time - INTERVAL '180 days'
       AND c.cxr_time <  e.echo_time
    WHERE BTRIM(COALESCE(c.view_position, '')) = 'AP'
    GROUP BY
        e.subject_id,
        e.measurement_id
)

SELECT
    e.subject_id,
    e.measurement_id,
    e.echo_time,
    g.ecg_n,
    c.ap_cxr_n
FROM echo e
JOIN ecg_count g
    ON g.subject_id = e.subject_id
   AND g.measurement_id = e.measurement_id
JOIN ap_cxr_count c
    ON c.subject_id = e.subject_id
   AND c.measurement_id = e.measurement_id
WHERE g.ecg_n >= 2
  AND c.ap_cxr_n >= 2;
"""

with conn.cursor() as cur:
    cur.execute(create_target_sql)

    cur.execute("""
        CREATE INDEX idx_tmp_echo_targets_subject
        ON tmp_echo_targets(subject_id);
    """)

    cur.execute("""
        CREATE INDEX idx_tmp_echo_targets_measurement
        ON tmp_echo_targets(measurement_id);
    """)

    cur.execute("""
        CREATE INDEX idx_tmp_echo_targets_subject_time
        ON tmp_echo_targets(subject_id, echo_time);
    """)

conn.commit()


# ============================================================
# 5. LOAD TARGETS
# ============================================================

target_columns, target_rows = fetch_all(
    conn,
    """
    SELECT
        subject_id,
        measurement_id,
        echo_time,
        ecg_n,
        ap_cxr_n
    FROM tmp_echo_targets
    ORDER BY subject_id, echo_time, measurement_id;
    """
)

num_samples = len(target_rows)
num_patients = len({row[0] for row in target_rows})

print(f"Target samples : {num_samples:,}")
print(f"Target patients: {num_patients:,}")
print(f"Expected       : ~{EXPECTED_SAMPLES:,} samples / ~{EXPECTED_PATIENTS:,} patients")

if num_samples != EXPECTED_SAMPLES or num_patients != EXPECTED_PATIENTS:
    print(
        "WARNING: cohort size differs from the previous SQL result. "
        "Check timestamp semantics / database contents before training."
    )


# ============================================================
# 6. INITIALIZE SAMPLE OBJECTS
# ============================================================

samples = {}

for subject_id, measurement_id, echo_time, ecg_n, ap_cxr_n in target_rows:
    key = (int(subject_id), int(measurement_id))
    sample_id = f"{subject_id}_{measurement_id}"

    samples[key] = {
        "sample_id": sample_id,
        "subject_id": int(subject_id),
        "measurement_id": int(measurement_id),
        "echo_time": echo_time,

        # cohort-count metadata for debugging/reproducibility
        "cohort_counts": {
            "ecg_studies_180d": int(ecg_n),
            "ap_cxr_studies_180d": int(ap_cxr_n),
        },

        "ecg": [],
        "cxr": [],
        "echo": {},
    }


# ============================================================
# 7. LOAD ECG HISTORY FROM ecg_master
# ============================================================

print("\nLoading ECG history from echo_multimodal.ecg_master...")

ecg_sql = r"""
SELECT
    t.subject_id,
    t.measurement_id,
    g.ecg_study_id,
    g.ecg_time,
    g.file_name,
    g.ecg_path
FROM tmp_echo_targets t
JOIN echo_multimodal.ecg_master g
    ON g.subject_id = t.subject_id
   AND g.ecg_time >= t.echo_time - INTERVAL '180 days'
   AND g.ecg_time <  t.echo_time
ORDER BY
    t.subject_id,
    t.measurement_id,
    g.ecg_time,
    g.ecg_study_id;
"""

_, ecg_rows = fetch_all(conn, ecg_sql)
print(f"ECG rows: {len(ecg_rows):,}")

for (
    subject_id,
    measurement_id,
    ecg_study_id,
    ecg_time,
    file_name,
    ecg_path,
) in tqdm(ecg_rows, desc="Building ECG lists"):

    key = (int(subject_id), int(measurement_id))

    samples[key]["ecg"].append({
        "study_id": int(ecg_study_id),
        "time": ecg_time,
        "path": resolve_dataset_path(ECG_ROOT, ecg_path),
        "file_name": file_name,
    })


# ============================================================
# 8. LOAD AP CXR HISTORY FROM cxr_master
# ============================================================
#
# Important:
#   cxr_master is IMAGE-level, so one cxr_study_id can have >1 AP image.
#   For temporal modeling, this version defines:
#       1 CXR study = 1 temporal event = 1 AP image
#
#   ROW_NUMBER() deterministically selects one AP image per study.
#
# ============================================================

print("\nLoading AP CXR history from echo_multimodal.cxr_master...")

cxr_sql = r"""
WITH ap_one_image_per_study AS (
    SELECT
        c.subject_id,
        c.cxr_study_id,
        c.dicom_id,
        c.cxr_time,
        c.view_position,
        c.performed_procedure_step_description,
        c.image_path,
        ROW_NUMBER() OVER (
            PARTITION BY c.subject_id, c.cxr_study_id
            ORDER BY c.dicom_id
        ) AS rn
    FROM echo_multimodal.cxr_master c
    WHERE BTRIM(COALESCE(c.view_position, '')) = 'AP'
)

SELECT
    t.subject_id,
    t.measurement_id,
    c.cxr_study_id,
    c.dicom_id,
    c.cxr_time,
    c.view_position,
    c.performed_procedure_step_description,
    c.image_path
FROM tmp_echo_targets t
JOIN ap_one_image_per_study c
    ON c.subject_id = t.subject_id
   AND c.rn = 1
   AND c.cxr_time >= t.echo_time - INTERVAL '180 days'
   AND c.cxr_time <  t.echo_time
ORDER BY
    t.subject_id,
    t.measurement_id,
    c.cxr_time,
    c.cxr_study_id;
"""

_, cxr_rows = fetch_all(conn, cxr_sql)
print(f"AP CXR study rows (1 image/study): {len(cxr_rows):,}")

for (
    subject_id,
    measurement_id,
    cxr_study_id,
    dicom_id,
    cxr_time,
    view_position,
    performed_procedure_step_description,
    image_path,
) in tqdm(cxr_rows, desc="Building AP CXR lists"):

    key = (int(subject_id), int(measurement_id))

    samples[key]["cxr"].append({
        "study_id": int(cxr_study_id),
        "dicom_id": dicom_id,
        "time": cxr_time,
        "view_position": view_position,
        "performed_procedure_step_description": performed_procedure_step_description,
        "path": resolve_dataset_path(CXR_ROOT, image_path),
    })


# ============================================================
# 9. LOAD ALL NON-EMPTY ECHO STRUCTURED LABELS
# ============================================================

print("\nLoading Echo labels...")

echo_sql = r"""
SELECT
    t.subject_id,
    t.measurement_id,
    s.measurement,
    s.measurement_description,
    s.result,
    s.unit
FROM tmp_echo_targets t
JOIN mimiciv_echo.structured_measurement s
    ON s.subject_id = t.subject_id
   AND s.measurement_id = t.measurement_id
WHERE s.result IS NOT NULL
  AND BTRIM(s.result) <> ''
ORDER BY
    t.subject_id,
    t.measurement_id,
    s.measurement;
"""

_, echo_rows = fetch_all(conn, echo_sql)
print(f"Echo label rows: {len(echo_rows):,}")

for (
    subject_id,
    measurement_id,
    measurement,
    description,
    result,
    unit,
) in tqdm(echo_rows, desc="Building Echo labels"):

    key = (int(subject_id), int(measurement_id))

    # Keep raw result as text because some Echo targets are categorical
    # (e.g., Normal / Mild / Moderate / Severe), not purely numeric.
    samples[key]["echo"][measurement] = {
        "result": result,
        "unit": unit,
        "description": description,
    }


# ============================================================
# 10. SORT LONGITUDINAL SEQUENCES
# ============================================================

sample_list = list(samples.values())

print("\nSorting ECG/CXR sequences chronologically...")
for sample in tqdm(sample_list, desc="Sorting"):
    sample["ecg"].sort(key=lambda x: x["time"])
    sample["cxr"].sort(key=lambda x: x["time"])


# ============================================================
# 11. QUALITY CHECKS
# ============================================================

print("\nRunning quality checks...")

bad_ecg = 0
bad_cxr = 0
bad_non_ap = 0
bad_duplicate_cxr_study = 0
bad_ecg_time = 0
bad_cxr_time = 0
missing_ecg_path = 0
missing_cxr_path = 0

history_seconds = 180 * 24 * 3600

for sample in sample_list:
    echo_time = sample["echo_time"]

    ecg_studies = {x["study_id"] for x in sample["ecg"]}
    cxr_studies = {x["study_id"] for x in sample["cxr"]}

    if len(ecg_studies) < 2:
        bad_ecg += 1

    if len(cxr_studies) < 2:
        bad_cxr += 1

    if any((x.get("view_position") or "").strip() != "AP" for x in sample["cxr"]):
        bad_non_ap += 1

    if len(cxr_studies) != len(sample["cxr"]):
        bad_duplicate_cxr_study += 1

    for x in sample["ecg"]:
        if not (x["time"] < echo_time):
            bad_ecg_time += 1

        # Existence check is informative, not used to alter the cohort.
        p = x.get("path")
        if p and not Path(p).with_suffix(".hea").exists() and not Path(p).exists():
            missing_ecg_path += 1

    for x in sample["cxr"]:
        # cxr_time is timestamptz in cxr_master, while echo_time is timestamp.
        # PostgreSQL has already enforced the cohort window in SQL.
        # Here we avoid Python timezone-naive/aware comparison problems.
        p = x.get("path")
        if p and not Path(p).exists():
            missing_cxr_path += 1

print(f"samples                         : {len(sample_list):,}")
print(f"patients                        : {len({s['subject_id'] for s in sample_list}):,}")
print(f"samples with <2 ECG studies     : {bad_ecg}")
print(f"samples with <2 AP CXR studies  : {bad_cxr}")
print(f"samples containing non-AP CXR   : {bad_non_ap}")
print(f"samples with duplicate CXR study: {bad_duplicate_cxr_study}")
print(f"invalid ECG time events          : {bad_ecg_time}")
print(f"missing ECG paths (informative)  : {missing_ecg_path:,}")
print(f"missing CXR paths (informative)  : {missing_cxr_path:,}")

assert bad_ecg == 0, "Cohort violation: some samples have <2 ECG studies"
assert bad_cxr == 0, "Cohort violation: some samples have <2 AP CXR studies"
assert bad_non_ap == 0, "Cohort violation: non-AP CXR found"
assert bad_duplicate_cxr_study == 0, "CXR extraction violation: >1 image retained per study"
assert bad_ecg_time == 0, "Temporal leakage: ECG at/after Echo found"


# ============================================================
# 12. SAVE TRAINING-FRIENDLY PKL
# ============================================================

print(f"\nSaving PKL: {OUTPUT_PKL}")
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(sample_list, f, protocol=pickle.HIGHEST_PROTOCOL)


# ============================================================
# 13. SAVE FULL HUMAN-READABLE JSONL
# ============================================================
#
# JSONL = one JSON object per line.
# Easier to inspect/stream than one gigantic pretty-printed JSON array.
#
# ============================================================

print(f"Saving JSONL: {OUTPUT_JSONL}")
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for sample in sample_list:
        f.write(
            json.dumps(
                sample,
                ensure_ascii=False,
                default=json_default,
            )
            + "\n"
        )


# Pretty preview for manual inspection
print(f"Saving preview JSON: {PREVIEW_JSON}")
with open(PREVIEW_JSON, "w", encoding="utf-8") as f:
    json.dump(
        sample_list[:100],
        f,
        indent=2,
        ensure_ascii=False,
        default=json_default,
    )


# ============================================================
# 14. SAVE DATASET METADATA
# ============================================================

meta = {
    "dataset_name": "MIMIC longitudinal ECG + AP-CXR -> TTE Echo phenotype",
    "sample_anchor": "one TTE Echo measurement_id",
    "history_days": 180,
    "echo_type": "tte",
    "ecg_source": "echo_multimodal.ecg_master",
    "cxr_source": "echo_multimodal.cxr_master",
    "patient_pool": "echo_multimodal.echo_patients",
    "echo_label_source": "mimiciv_echo.structured_measurement",
    "min_ecg_studies": 2,
    "cxr_view": "AP",
    "min_ap_cxr_studies": 2,
    "one_cxr_image_per_study": True,
    "cxr_image_selection_rule": "first AP image ordered by dicom_id",
    "history_interval": "[echo_time - 180 days, echo_time)",
    "recent_anchor_hours": None,
    "duplicate_echo_filter": False,
    "minimum_target_gap_days": None,
    "hadm_restriction": False,
    "split_requirement": "subject-level split",
    "num_samples": len(sample_list),
    "num_patients": len({s["subject_id"] for s in sample_list}),
    "expected_num_samples": EXPECTED_SAMPLES,
    "expected_num_patients": EXPECTED_PATIENTS,
}

print(f"Saving metadata: {META_FILE}")
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)


conn.close()

print("\nDone.")
print(f"PKL      : {OUTPUT_PKL}")
print(f"JSONL    : {OUTPUT_JSONL}")
print(f"Preview  : {PREVIEW_JSON}")
print(f"Metadata : {META_FILE}")
