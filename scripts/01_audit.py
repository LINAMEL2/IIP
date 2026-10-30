"""Phase 1 - Step 1: data audit and validation for the COCO subset.

Checks: missing files, corrupt images, exact duplicates, captions per image.
Writes audit_report.json and clean_image_ids.json (nothing is deleted on disk).

Usage:
    python 01_audit.py --images data/images --annotations data/captions.json --out outputs
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

EXPECTED_CAPTIONS = 5


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out", default="outputs", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, encoding="utf-8") as f:
        ann = json.load(f)

    images = {im["id"]: im for im in ann["images"]}
    captions = defaultdict(list)
    orphan_captions = []
    for a in ann["annotations"]:
        if a["image_id"] in images:
            captions[a["image_id"]].append(a["caption"])
        else:
            orphan_captions.append(a["id"])

    missing, corrupt = [], []
    hashes = defaultdict(list)
    for img_id, im in images.items():
        path = args.images / im["file_name"]
        if not path.exists():
            missing.append(img_id)
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:  # verify() leaves the file unusable
                img.convert("RGB").load()
        except Exception as e:  # unreadable or truncated file
            corrupt.append({"image_id": img_id, "error": str(e)})
            continue
        hashes[md5(path)].append(img_id)

    # Duplicates: keep the lowest image id of each group, drop the others
    duplicate_groups = [sorted(ids) for ids in hashes.values() if len(ids) > 1]
    dropped_duplicates = sorted(i for g in duplicate_groups for i in g[1:])

    # Caption count: flagged if != 5; removed only if there is no caption at all
    caption_count_issues = {
        i: len(captions.get(i, []))
        for i in images
        if len(captions.get(i, [])) != EXPECTED_CAPTIONS
    }
    no_caption = [i for i, n in caption_count_issues.items() if n == 0]

    removed = (
        set(missing)
        | {c["image_id"] for c in corrupt}
        | set(dropped_duplicates)
        | set(no_caption)
    )
    clean_ids = sorted(set(images) - removed)

    report = {
        "n_images_in_annotations": len(images),
        "n_images_clean": len(clean_ids),
        "missing_files": missing,
        "corrupt_files": corrupt,
        "duplicate_groups": duplicate_groups,
        "dropped_duplicates": dropped_duplicates,
        "caption_count_issues": caption_count_issues,
        "images_without_caption": no_caption,
        "orphan_captions": orphan_captions,
    }
    with open(args.out / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(args.out / "clean_image_ids.json", "w", encoding="utf-8") as f:
        json.dump(clean_ids, f)

    print(f"Images in annotations : {len(images)}")
    print(f"Missing files         : {len(missing)}")
    print(f"Corrupt files         : {len(corrupt)}")
    print(f"Duplicate groups      : {len(duplicate_groups)} ({len(dropped_duplicates)} dropped)")
    print(f"Caption count != {EXPECTED_CAPTIONS}    : {len(caption_count_issues)}")
    print(f"Clean images kept     : {len(clean_ids)}")


if __name__ == "__main__":
    main()
