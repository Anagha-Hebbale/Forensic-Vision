import json
import os
from datetime import datetime, timezone
from typing import Dict, List

from PIL import Image


DATASET_ROOT = "dataset"
REPORT_PATH = "models/data_audit_report.json"
SPLITS = ["train", "valid", "test"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_image_files(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(folder):
        for n in names:
            ext = os.path.splitext(n.lower())[1]
            if ext in VALID_EXTS:
                files.append(os.path.join(root, n))
    return files


def verify_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def audit_split(split_dir: str) -> Dict[str, object]:
    if not os.path.isdir(split_dir):
        return {"exists": False, "classes": {}, "total": 0}

    class_report: Dict[str, Dict[str, object]] = {}
    total = 0
    for name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, name)
        if not os.path.isdir(class_dir):
            continue

        files = iter_image_files(class_dir)
        corrupted = [p for p in files if not verify_image(p)]
        ok_count = len(files) - len(corrupted)
        total += ok_count

        class_report[name] = {
            "valid_images": ok_count,
            "corrupted_images": len(corrupted),
            "sample_corrupted": corrupted[:5],
        }

    return {"exists": True, "classes": class_report, "total": total}


def class_balance(classes: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    if not classes:
        return {"ratio_max_to_min": None, "warning": "No class folders found."}

    counts = [int(v["valid_images"]) for v in classes.values()]
    min_count = min(counts)
    max_count = max(counts)
    ratio = float(max_count / min_count) if min_count > 0 else None

    warning = ""
    if ratio is None:
        warning = "At least one class has zero valid images."
    elif ratio > 1.5:
        warning = "Class imbalance detected (max/min > 1.5)."

    return {"ratio_max_to_min": ratio, "warning": warning}


def main() -> None:
    os.makedirs("models", exist_ok=True)
    payload: Dict[str, object] = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": DATASET_ROOT,
        "splits": {},
    }

    for split in SPLITS:
        split_dir = os.path.join(DATASET_ROOT, split)
        report = audit_split(split_dir)
        if report.get("exists"):
            report["balance"] = class_balance(report.get("classes", {}))
        payload["splits"][split] = report

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Data audit complete")
    print(f"Saved report -> {REPORT_PATH}")
    for split, info in payload["splits"].items():
        print(f"[{split}] exists={info.get('exists')} total={info.get('total')}")


if __name__ == "__main__":
    main()
