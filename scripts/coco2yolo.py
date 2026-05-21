"""
Convert ClearSAR COCO annotations to YOLO format.
Also creates 5-fold cross-validation splits.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

DATA_DIR = Path("data/ClearSAR/data")
YOLO_DIR = Path("data/yolo")
N_FOLDS = 5
SEED = 42


def _link(src: Path, dst: Path):
    """Create a relative symlink dst -> src, replacing if present."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def main():
    anno_path = DATA_DIR / "annotations/instances_train.json"
    anno = json.loads(anno_path.read_text())

    imgs = {i["id"]: i for i in anno["images"]}
    # Group annotations by image_id
    ann_by_img = {}
    for a in anno["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    # Output dirs
    labels_dir = YOLO_DIR / "labels/all"
    labels_dir.mkdir(parents=True, exist_ok=True)

    img_ids = list(imgs.keys())

    # Write label files
    print(f"Writing {len(img_ids)} label files...")
    for img_id in img_ids:
        img = imgs[img_id]
        w, h = img["width"], img["height"]
        anns = ann_by_img.get(img_id, [])
        label_path = labels_dir / f"{img_id}.txt"
        lines = []
        for a in anns:
            cx, cy, nw, nh = coco_to_yolo(a["bbox"], w, h)
            # Clamp to [0,1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(1e-5, min(1.0, nw))
            nh = max(1e-5, min(1.0, nh))
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        label_path.write_text("\n".join(lines))

    # 5-fold splits
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    img_ids_arr = np.array(img_ids)
    img_dir = DATA_DIR / "images/train"
    MIN_VALID = 1000  # real PNGs are much larger than presigned-URL JSON stubs

    for fold, (train_idx, val_idx) in enumerate(kf.split(img_ids_arr)):
        fold_dir = YOLO_DIR / f"fold{fold}"
        for split, indices in [("train", train_idx), ("val", val_idx)]:
            (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            for idx in indices:
                img_id = img_ids_arr[idx]
                img_src = img_dir / f"{img_id}.png"
                lbl_src = labels_dir / f"{img_id}.txt"
                if img_src.exists() and img_src.stat().st_size > MIN_VALID:
                    _link(img_src, fold_dir / "images" / split / f"{img_id}.png")
                    _link(lbl_src, fold_dir / "labels" / split / f"{img_id}.txt")
        print(f"  Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")

    # Test images (no labels)
    test_dir = DATA_DIR / "images/test"
    yolo_test_dir = YOLO_DIR / "test/images"
    yolo_test_dir.mkdir(parents=True, exist_ok=True)
    test_imgs = list(test_dir.glob("*.png"))
    for src in test_imgs:
        dst = yolo_test_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"Test images: {len(test_imgs)}")

    # Write dataset YAML for each fold
    for fold in range(N_FOLDS):
        fold_dir = YOLO_DIR / f"fold{fold}"
        yaml_content = f"""path: {fold_dir.resolve()}
train: images/train
val: images/val

nc: 1
names: ['RFI']
"""
        (fold_dir / "dataset.yaml").write_text(yaml_content)

    print("Done.")


if __name__ == "__main__":
    # Run from the repo root so the hardcoded data/ paths resolve.
    os.chdir(Path(__file__).parent.parent)
    main()
