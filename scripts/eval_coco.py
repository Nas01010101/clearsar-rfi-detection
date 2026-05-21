#!/usr/bin/env python3
"""Score COCO-format predictions against COCO-format GT.

Outputs: mAP50-95, mAP50, mAP75, AP_small/medium/large
"""
import argparse
import json
import os
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=None, help="optional JSON to write summary to")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io

    raw_preds = json.load(open(args.preds))
    preds = []
    for det in raw_preds:
        bbox = det.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        preds.append(det)

    gt = COCO(args.gt)
    if not preds:
        stats = {
            "mAP50-95": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
        }
        print(json.dumps(stats, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(stats, f, indent=2)
        return

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(preds, f)
            tmp_name = f.name
        dt = gt.loadRes(tmp_name)
        e = COCOeval(gt, dt, "bbox")
        with contextlib.redirect_stdout(io.StringIO()):
            e.evaluate()
            e.accumulate()
            e.summarize()
    finally:
        if tmp_name:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)

    stats = {
        "mAP50-95": float(e.stats[0]),
        "mAP50":    float(e.stats[1]),
        "mAP75":    float(e.stats[2]),
        "AP_small": float(e.stats[3]),
        "AP_medium":float(e.stats[4]),
        "AP_large": float(e.stats[5]),
        "AR_1":     float(e.stats[6]),
        "AR_10":    float(e.stats[7]),
        "AR_100":   float(e.stats[8]),
        "AR_small": float(e.stats[9]),
        "AR_medium":float(e.stats[10]),
        "AR_large": float(e.stats[11]),
    }
    print(json.dumps(stats, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
