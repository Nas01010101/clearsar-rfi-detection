#!/usr/bin/env python
"""Reproduce the leak-isolated meta-validation ablation chain for the ICIP
camera-ready revision (reviewer TP-1018).

Each pipeline stage is scored on the 401-image meta-validation set with
pycocotools, from the saved per-fold / per-resolution prediction JSONs. The
point of the script is to (a) verify the numbers cited in the paper and
(b) make the per-component chain reproducible from the archived predictions.

Usage:
    python scripts/verify_ablation.py --archive <path-to-ClearSAT-T1-v2-archive>
"""
import argparse
import json
import os
import sys
import tempfile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    weighted_boxes_fusion = None


def coco_eval(gt_path, dets):
    """Full COCOeval stats vector for a flat COCO detection list.

    Returns a dict with the size/IoU-stratified averages reviewers expect:
    AP@[.50:.95], AP50, AP75, and AP for small/medium/large areas.
    """
    gt = COCO(gt_path)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(dets, f)
        tmp = f.name
    try:
        dt = gt.loadRes(tmp)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        s = ev.stats
        return {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
                "AP_s": float(s[3]), "AP_m": float(s[4]), "AP_l": float(s[5])}
    finally:
        os.unlink(tmp)


def coco_map(gt_path, dets):
    """mAP@[.50:.95] only (back-compat helper)."""
    return coco_eval(gt_path, dets)["AP"]


def load(path):
    with open(path) as f:
        return json.load(f)


def wbf_fuse(pred_lists, gt_path, iou_thr=0.70, weights=None, conf_type="max"):
    """Weighted Box Fusion over several flat COCO detection lists.

    WBF needs normalized [0,1] coords, so we scale by each image's true size
    taken from the GT (the meta_val GT carries width/height per image).
    """
    if weighted_boxes_fusion is None:
        raise RuntimeError("pip install ensemble-boxes")
    gt = COCO(gt_path)
    sizes = {im["id"]: (im["width"], im["height"]) for im in gt.dataset["images"]}
    img_ids = sorted(sizes)

    # group each list's dets by image
    grouped = []
    for preds in pred_lists:
        g = {}
        for d in preds:
            g.setdefault(d["image_id"], []).append(d)
        grouped.append(g)

    fused = []
    for iid in img_ids:
        W, H = sizes[iid]
        boxes_l, scores_l, labels_l = [], [], []
        for g in grouped:
            bs, ss, ls = [], [], []
            for d in g.get(iid, []):
                x, y, w, h = d["bbox"]
                bs.append([x / W, y / H, (x + w) / W, (y + h) / H])
                ss.append(d["score"])
                ls.append(0)
            boxes_l.append(bs)
            scores_l.append(ss)
            labels_l.append(ls)
        if not any(boxes_l):
            continue
        b, s, _ = weighted_boxes_fusion(
            boxes_l, scores_l, labels_l,
            weights=weights, iou_thr=iou_thr, conf_type=conf_type,
        )
        for (x1, y1, x2, y2), sc in zip(b, s):
            fused.append({
                "image_id": iid, "category_id": 0,
                "bbox": [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H],
                "score": float(sc),
            })
    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True,
                    help="path to ClearSAT-T1-v2 archive root")
    args = ap.parse_args()
    A = args.archive
    gt = os.path.join(A, "cpu_eval/gt/meta_val_coco.json")
    pf = os.path.join(A, "cpu_eval/v2_pseudo_meta_perfold")

    if not os.path.exists(gt):
        sys.exit(f"meta_val GT not found at {gt}")

    rows = []

    # Stage: pseudo-FT student, fold 4, single resolution 640
    f4_640 = load(os.path.join(pf, "fold4_sz640.json"))
    rows.append(("Pseudo-FT student (fold 4) @640", coco_eval(gt, f4_640)))

    # Stage: + multi-resolution TTA (608/640/672/704) WBF, fold 4
    f4_tta = wbf_fuse(
        [load(os.path.join(pf, f"fold4_sz{r}.json")) for r in (608, 640, 672, 704)],
        gt,
    )
    rows.append(("  + multi-res TTA (4 res, WBF)", coco_eval(gt, f4_tta)))

    # Stage: + 3-fold cross-fold WBF dropping fold 0 (folds 1,2,4), each TTA-fused
    per_fold_tta = []
    for fold in (1, 2, 4):
        ft = wbf_fuse(
            [load(os.path.join(pf, f"fold{fold}_sz{r}.json")) for r in (608, 640, 672, 704)],
            gt,
        )
        per_fold_tta.append(ft)
    super3 = wbf_fuse(per_fold_tta, gt)
    rows.append(("  + 3-fold WBF (drop fold 0) = S*", coco_eval(gt, super3)))

    print("\n==== Leak-isolated meta-validation ablation ====")
    print(f"  {'Configuration':36s} {'AP':>6s} {'AP50':>6s} {'AP75':>6s} "
          f"{'AP_s':>6s} {'AP_m':>6s} {'AP_l':>6s}")
    prev = None
    for name, m in rows:
        delta = "" if prev is None else f"  (dAP {m['AP'] - prev:+.4f})"
        print(f"  {name:36s} {m['AP']:.4f} {m['AP50']:.4f} {m['AP75']:.4f} "
              f"{m['AP_s']:.4f} {m['AP_m']:.4f} {m['AP_l']:.4f}{delta}")
        prev = m["AP"]

    saved = os.path.join(A, "cpu_eval/v2_pseudo_ft_TTA_3fold_drop0_meta_val.eval.json")
    if os.path.exists(saved):
        s = load(saved)
        print(f"\n  Archived S* eval (reference): AP={s['mAP50-95']:.4f} "
              f"AP50={s['mAP50']:.4f} AP75={s['mAP75']:.4f} "
              f"AP_s={s['AP_small']:.4f} AP_m={s['AP_medium']:.4f} AP_l={s['AP_large']:.4f}")


if __name__ == "__main__":
    main()
