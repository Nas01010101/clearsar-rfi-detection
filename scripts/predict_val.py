"""
Run one fold's weights on another fold's val set (for leaked cross-fold WBF eval).
"""
import argparse, json, time
from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--val-dir', required=True, help='fold{k}/images/val')
    ap.add_argument('--out', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf',  type=float, default=0.001)
    ap.add_argument('--cat-id', type=int, default=0, help='0 for fold val GT compatibility')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = YOLO(args.weights)
    imgs = sorted(Path(args.val_dir).glob('*.png'), key=lambda p: int(p.stem))
    print(f'[predict_val] {len(imgs)} images, device={device}', flush=True)
    dets = []
    t0 = time.time()
    for i in range(0, len(imgs), 16):
        b = imgs[i:i+16]
        r = model.predict([str(p) for p in b], imgsz=args.imgsz, conf=args.conf,
                          device=device, verbose=False, augment=False, max_det=300)
        for p, res in zip(b, r):
            iid = int(p.stem)
            boxes = res.boxes
            if boxes is None or len(boxes)==0: continue
            xyxy = boxes.xyxy.cpu().numpy(); scores = boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), s in zip(xyxy, scores):
                dets.append({'image_id': iid, 'category_id': args.cat_id,
                             'bbox': [float(x1),float(y1),float(x2-x1),float(y2-y1)],
                             'score': float(s)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,'w') as f: json.dump(dets, f)
    print(f'[predict_val] wrote {len(dets)} dets in {time.time()-t0:.1f}s', flush=True)

if __name__=='__main__': main()
