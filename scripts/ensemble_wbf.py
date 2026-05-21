"""
Weighted Box Fusion ensemble for COCO-format detection JSONs.

Two modes:
  1. Single-stage WBF over N inputs:
       python ensemble_wbf.py \
         --inputs f0.json f1.json f2.json f3.json f4.json \
         --weights 1.0 1.0 1.0 1.0 1.5 \
         --iou 0.70 --conf-type max \
         --img-dir data/ClearSAR/data/images/test \
         --out submission.json

  2. Two-stage (the v4 winner — peer consolidation then anchor refinement):
       python ensemble_wbf.py --multistage \
         --stage1 f2.json f3.json --stage1-iou 0.55 --stage1-conf avg \
         --stage2-anchor f4.json --stage2-anchor-weight 1.5 \
         --stage2-iou 0.70 --stage2-conf max \
         --img-dir data/ClearSAR/data/images/test \
         --out submission.json

The output is a flat COCO detections list:
  [{image_id, category_id:1, bbox:[x,y,w,h], score}, ...]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from ensemble_boxes import weighted_boxes_fusion


def load_dims(img_dir: str) -> dict:
    """Return {image_id: (W, H)} by reading actual test image files."""
    dims = {}
    for p in Path(img_dir).glob("*.png"):
        try:
            iid = int(p.stem)
        except ValueError:
            continue
        with Image.open(p) as im:
            dims[iid] = im.size
    if not dims:
        raise ValueError(f"No images found in {img_dir}")
    return dims


def group_by_image(detections):
    by = defaultdict(list)
    for d in detections:
        by[int(d["image_id"])].append(d)
    return by


def to_wbf(preds, W, H):
    B, S, L = [], [], []
    for p in preds:
        x, y, w, h = p["bbox"]
        x1 = max(0, min(1, x / W));  y1 = max(0, min(1, y / H))
        x2 = max(0, min(1, (x + w) / W));  y2 = max(0, min(1, (y + h) / H))
        if x2 <= x1 or y2 <= y1:
            continue
        B.append([x1, y1, x2, y2])
        S.append(float(p["score"]))
        L.append(0)  # class-agnostic for single-class
    return B, S, L


def from_wbf(B, S, L, iid, W, H, cat_id=1):
    return [
        {
            "image_id": iid,
            "category_id": cat_id,
            "bbox": [b[0] * W, b[1] * H, (b[2] - b[0]) * W, (b[3] - b[1]) * H],
            "score": float(s),
        }
        for b, s, l in zip(B, S, L)
    ]


def single_stage(preds_list, weights, dims, iou_thr, conf_type, skip_box_thr=0.0):
    by_lists = [group_by_image(p) for p in preds_list]
    out = []
    for iid, (W, H) in dims.items():
        lists = [to_wbf(by.get(iid, []), W, H) for by in by_lists]
        Bs, Ss, Ls = zip(*lists) if lists else ([], [], [])
        if not any(len(b) for b in Bs):
            continue
        B, S, L = weighted_boxes_fusion(
            list(Bs), list(Ss), list(Ls),
            weights=weights, iou_thr=iou_thr,
            skip_box_thr=skip_box_thr, conf_type=conf_type,
        )
        out += from_wbf(B, S, L, iid, W, H)
    return out


def multistage(stage1_preds, stage1_weights, stage1_iou, stage1_conf,
               stage2_anchor_preds, stage2_anchor_weight,
               stage2_iou, stage2_conf, dims):
    """Two-stage WBF — Stage 1 fuses peers, Stage 2 anchors against an extra model."""
    s1_by = [group_by_image(p) for p in stage1_preds]
    s2_anchor_by = group_by_image(stage2_anchor_preds)
    out = []
    for iid, (W, H) in dims.items():
        # Stage 1: fuse peer models
        s1_lists = [to_wbf(by.get(iid, []), W, H) for by in s1_by]
        Bs, Ss, Ls = zip(*s1_lists) if s1_lists else ([], [], [])
        if any(len(b) for b in Bs):
            B1, S1, L1 = weighted_boxes_fusion(
                list(Bs), list(Ss), list(Ls),
                weights=stage1_weights, iou_thr=stage1_iou,
                skip_box_thr=0.0, conf_type=stage1_conf,
            )
        else:
            B1, S1, L1 = [], [], []

        # Stage 2: anchor refinement
        Ba, Sa, La = to_wbf(s2_anchor_by.get(iid, []), W, H)
        if not (len(B1) or len(Ba)):
            continue
        B, S, L = weighted_boxes_fusion(
            [list(B1), Ba], [list(S1), Sa], [list(L1), La],
            weights=[1.0, stage2_anchor_weight], iou_thr=stage2_iou,
            skip_box_thr=0.0, conf_type=stage2_conf,
        )
        out += from_wbf(B, S, L, iid, W, H)
    return out


def cap_top_k_per_image(preds, k=100):
    by = group_by_image(preds)
    out = []
    for iid, ps in by.items():
        ps.sort(key=lambda x: x["score"], reverse=True)
        out.extend(ps[:k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multistage", action="store_true",
                    help="Use two-stage WBF (v4 winning recipe).")
    ap.add_argument("--inputs", nargs="*",
                    help="Single-stage: list of COCO-format detection JSONs to fuse.")
    ap.add_argument("--weights", nargs="*", type=float,
                    help="Single-stage: matching list of fold weights.")
    ap.add_argument("--iou", type=float, default=0.70,
                    help="Single-stage IoU threshold (hyperopt-optimal: 0.70).")
    ap.add_argument("--conf-type", default="max",
                    choices=["max", "avg", "box_and_model_avg", "absent_model_aware_avg"],
                    help="Single-stage WBF conf_type (default: max).")
    ap.add_argument("--skip", type=float, default=0.0,
                    help="skip_box_thr for low-conf boxes (default: 0.0).")
    # Multistage
    ap.add_argument("--stage1", nargs="*", help="Stage 1 inputs (peer models).")
    ap.add_argument("--stage1-weights", nargs="*", type=float,
                    help="Stage 1 weights (default: equal).")
    ap.add_argument("--stage1-iou", type=float, default=0.55)
    ap.add_argument("--stage1-conf", default="avg")
    ap.add_argument("--stage2-anchor", help="Stage 2 anchor model (single file).")
    ap.add_argument("--stage2-anchor-weight", type=float, default=1.5,
                    help="Anchor weight in stage 2 (v4 default: 1.5).")
    ap.add_argument("--stage2-iou", type=float, default=0.70)
    ap.add_argument("--stage2-conf", default="max")
    # Common
    ap.add_argument("--img-dir", default=None,
                    help="Path to test images (used for actual W,H per image_id).")
    ap.add_argument("--fixed-wh", type=int, default=None,
                    help="Use fixed square image size instead of reading from disk (e.g. 512).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=100,
                    help="Cap per-image preds at top-K by score (COCO max_dets=100). "
                         "Set 0 to disable.")
    args = ap.parse_args()

    if args.fixed_wh is not None:
        # Build the image_id set from the union of input pred files; assign every id
        # the same WxH. Avoids reading any image from disk.
        sources = list(args.inputs or [])
        if args.stage1:
            sources.extend(args.stage1)
        if args.stage2_anchor:
            sources.append(args.stage2_anchor)
        ids = set()
        for p in sources:
            ids.update(d["image_id"] for d in json.loads(Path(p).read_text()))
        dims = {iid: (args.fixed_wh, args.fixed_wh) for iid in ids}
        print(f"[ensemble_wbf] fixed-wh={args.fixed_wh}, {len(dims)} image ids from preds")
    else:
        if not args.img_dir:
            ap.error("either --img-dir or --fixed-wh is required")
        dims = load_dims(args.img_dir)
        print(f"[ensemble_wbf] loaded dims for {len(dims)} test images")

    if args.multistage:
        assert args.stage1 and args.stage2_anchor, "multistage needs --stage1 and --stage2-anchor"
        s1_preds = [json.loads(Path(p).read_text()) for p in args.stage1]
        s1_w = args.stage1_weights or [1.0] * len(s1_preds)
        s2_anchor = json.loads(Path(args.stage2_anchor).read_text())
        out = multistage(
            s1_preds, s1_w, args.stage1_iou, args.stage1_conf,
            s2_anchor, args.stage2_anchor_weight,
            args.stage2_iou, args.stage2_conf, dims,
        )
        print(f"[ensemble_wbf] multistage: {len(s1_preds)} peers + 1 anchor → {len(out)} preds")
    else:
        assert args.inputs, "single-stage needs --inputs"
        preds_list = [json.loads(Path(p).read_text()) for p in args.inputs]
        weights = args.weights or [1.0] * len(preds_list)
        assert len(weights) == len(preds_list), "weights must match inputs"
        out = single_stage(preds_list, weights, dims, args.iou, args.conf_type, args.skip)
        print(f"[ensemble_wbf] single-stage: {len(preds_list)} models → {len(out)} preds")

    if args.top_k > 0:
        before = len(out)
        out = cap_top_k_per_image(out, k=args.top_k)
        print(f"[ensemble_wbf] capped top-{args.top_k}/image: {before} → {len(out)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"[ensemble_wbf] wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
