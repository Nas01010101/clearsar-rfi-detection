"""
Local MPS inference of one fold's weights on the 786 test images.
Writes COCO-format detections at conf>=0.001, imgsz=640.
"""
import argparse, json, sys, time
from pathlib import Path
from ultralytics import YOLO
import torch

def read_ids(path):
    ids = set()
    if path is None:
        return ids
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            ids.add(int(Path(line).stem))
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--test-dir', default='data/ClearSAR/data/images/test')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf',  type=float, default=0.001)
    ap.add_argument('--cat-id', type=int, default=1)
    ap.add_argument('--ids-file', default=None,
                    help='Optional newline file of numeric image ids/stems to predict. Nonmatching PNGs are skipped.')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[predict] weights={args.weights} device={device} imgsz={args.imgsz} conf={args.conf}', flush=True)

    model = YOLO(args.weights)
    ids = read_ids(args.ids_file)
    skipped_nonnumeric = 0
    test_imgs = []
    for p in Path(args.test_dir).glob('*.png'):
        try:
            iid = int(p.stem)
        except ValueError:
            skipped_nonnumeric += 1
            continue
        if ids and iid not in ids:
            continue
        test_imgs.append(p)
    test_imgs = sorted(test_imgs, key=lambda p: int(p.stem))
    print(
        f'[predict] {len(test_imgs)} images'
        f"{f' from ids-file={args.ids_file}' if args.ids_file else ''}"
        f"{f' (skipped {skipped_nonnumeric} nonnumeric PNGs)' if skipped_nonnumeric else ''}",
        flush=True,
    )

    detections = []
    t0 = time.time()
    # Batch inference for speed
    for i in range(0, len(test_imgs), 16):
        batch = test_imgs[i:i+16]
        results = model.predict([str(p) for p in batch], imgsz=args.imgsz, conf=args.conf,
                                device=device, verbose=False, augment=False, max_det=300)
        for p, res in zip(batch, results):
            iid = int(p.stem)
            boxes = res.boxes
            if boxes is None or len(boxes)==0: continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), s in zip(xyxy, scores):
                detections.append({'image_id': iid, 'category_id': args.cat_id,
                                   'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                                   'score': float(s)})
        if (i//16) % 10 == 0:
            print(f'[predict] {i+len(batch)}/{len(test_imgs)} elapsed={time.time()-t0:.1f}s', flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(detections, f)
    print(f'[predict] wrote {len(detections)} dets to {args.out} in {time.time()-t0:.1f}s', flush=True)

if __name__ == '__main__':
    main()
