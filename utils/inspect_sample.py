"""运行 python -m utils.inspect_sample 检查一个样本的实际文件。"""

import argparse
import json
from itertools import islice

from .readers import iter_samples, read_cxr, read_ecg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="dataset/processed/samples.jsonl")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.index < 0:
        parser.error("--index must be nonnegative")
    sample = next(islice(iter_samples(args.samples), args.index, None), None)
    if sample is None:
        parser.error("sample index out of range")
    summary = {"sample_id": sample["sample_id"], "ecg": [], "cxr": []}
    for event in sample["ecg"]:
        record = read_ecg(event["path"], missing="keep")
        summary["ecg"].append({
            "time": event["time"],
            "shape": list(record["signal"].shape),
            "fs": record["fs"],
            "duration_seconds": record["duration_seconds"],
            "lead_names": record["lead_names"],
            "units": record["units"],
            "invalid_values": int((~record["valid_mask"]).sum()),
        })
    for event in sample["cxr"]:
        record = read_cxr(event["path"], size=(224, 224))
        summary["cxr"].append({
            "time": event["time"],
            "shape": list(record["image"].shape),
            "original_size": record["original_size"],
        })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
