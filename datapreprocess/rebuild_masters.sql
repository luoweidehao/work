-- Rebuild the patient-level master tables used by process.py.
-- Scope: retain every ECG/CXR record for every patient with at least one TTE.
-- The final 180-day window and AP-view eligibility filters are applied later
-- when building the training cohort, not in these master tables.

BEGIN;

-- `cxr_time` is stored as timestamptz. Use the same session timezone when
-- rebuilding so comparisons with Echo timestamps remain reproducible.
SET LOCAL TIME ZONE 'Asia/Shanghai';

-- Intentional destructive replacement. Keeping all three tables in the same
-- transaction prevents process.py from observing a partially rebuilt dataset.
DROP TABLE IF EXISTS echo_multimodal.ecg_master;
DROP TABLE IF EXISTS echo_multimodal.cxr_master;
DROP TABLE IF EXISTS echo_multimodal.echo_patients;

-- Patient universe: all subjects with at least one valid TTE measurement.
CREATE TABLE echo_multimodal.echo_patients AS
SELECT DISTINCT
    s.subject_id::bigint AS subject_id
FROM mimiciv_echo.structured_measurement AS s
WHERE s.test_type = 'tte'
  AND s.subject_id IS NOT NULL;

-- ECG master: all ECG studies for the Echo patient universe, with source paths
-- left relative to the MIMIC-ECG dataset root for portability.
CREATE TABLE echo_multimodal.ecg_master AS
SELECT
    r.subject_id,
    r.study_id AS ecg_study_id,
    r.ecg_time,
    r.file_name,
    r.path AS ecg_path
FROM mimiciv_ecg.record_list AS r
JOIN echo_multimodal.echo_patients AS p
    ON p.subject_id = r.subject_id;

-- CXR master: image-level records for the same patient universe.
-- `split` confirms that the image belongs to the local MIMIC-CXR distribution.
-- MIMIC-CXR files use files/pXX/p<subject_id>/s<study_id>/<dicom_id>.jpg.
CREATE TABLE echo_multimodal.cxr_master AS
SELECT
    m.subject_id,
    m.study_id AS cxr_study_id,
    m.dicom_id,
    to_timestamp(
        m.study_date || lpad(split_part(m.study_time, '.', 1), 6, '0'),
        'YYYYMMDDHH24MISS'
    ) AS cxr_time,
    m.view_position,
    m.performed_procedure_step_description,
    'files/p' || left(m.subject_id::text, 2)
        || '/p' || m.subject_id::text
        || '/s' || m.study_id::text
        || '/' || m.dicom_id || '.jpg' AS image_path
FROM mimiciv_cxr.metadata AS m
JOIN mimiciv_cxr.split AS s
    ON s.dicom_id = m.dicom_id
JOIN echo_multimodal.echo_patients AS p
    ON p.subject_id = m.subject_id;

-- Constraints and lookup indexes required by the temporal joins in process.py.
CREATE UNIQUE INDEX idx_echo_patients_subject
    ON echo_multimodal.echo_patients(subject_id);

CREATE UNIQUE INDEX idx_ecg_master_study
    ON echo_multimodal.ecg_master(ecg_study_id);
CREATE INDEX idx_ecg_master_subject
    ON echo_multimodal.ecg_master(subject_id);
CREATE INDEX idx_ecg_master_subject_time
    ON echo_multimodal.ecg_master(subject_id, ecg_time);

CREATE UNIQUE INDEX idx_cxr_master_dicom
    ON echo_multimodal.cxr_master(dicom_id);
CREATE INDEX idx_cxr_master_study
    ON echo_multimodal.cxr_master(cxr_study_id);
CREATE INDEX idx_cxr_master_subject
    ON echo_multimodal.cxr_master(subject_id);
CREATE INDEX idx_cxr_master_subject_time
    ON echo_multimodal.cxr_master(subject_id, cxr_time);

-- Refresh planner statistics before the large 180-day cohort queries.
ANALYZE echo_multimodal.echo_patients;
ANALYZE echo_multimodal.ecg_master;
ANALYZE echo_multimodal.cxr_master;

COMMIT;

-- Post-rebuild audit: confirm table sizes and patient coverage.
SELECT 'echo_patients' AS table_name, COUNT(*) AS rows,
       COUNT(DISTINCT subject_id) AS subjects
FROM echo_multimodal.echo_patients
UNION ALL
SELECT 'ecg_master', COUNT(*), COUNT(DISTINCT subject_id)
FROM echo_multimodal.ecg_master
UNION ALL
SELECT 'cxr_master', COUNT(*), COUNT(DISTINCT subject_id)
FROM echo_multimodal.cxr_master;
