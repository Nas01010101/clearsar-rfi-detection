"""
Train YOLO26 on ClearSAR with MLflow tracking.
Usage: python scripts/train_yolo.py [--fold 0] [--model yolo26x.pt] [--epochs 100] [--imgsz 640]
"""
import argparse
import contextlib
import io
import importlib.util
import json
import os
import time
from pathlib import Path

import mlflow
import torch
from ultralytics import YOLO


def get_device():
    """Return Ultralytics-compatible device spec.

    Multi-GPU: "0,1" triggers DDP in Ultralytics.
    Single CUDA: "cuda". MPS for Mac local. CPU fallback.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return ",".join(str(i) for i in range(n)) if n > 1 else "cuda"
    return "cpu"


def get_predict_device():
    """Use one device for post-train inference even if training used DDP."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_coco(weights: Path, val_dir: Path, out_path: Path, imgsz: int, cat_id: int) -> int:
    """Run low-threshold validation inference and write COCO detections."""
    device = get_predict_device()
    model = YOLO(str(weights))
    imgs = sorted(val_dir.glob("*.png"), key=lambda p: int(p.stem))
    dets = []
    t0 = time.time()
    print(f"[eval] predicting {len(imgs)} val images on {device}", flush=True)
    for i in range(0, len(imgs), 16):
        batch = imgs[i:i + 16]
        results = model.predict(
            [str(p) for p in batch],
            imgsz=imgsz,
            conf=0.001,
            device=device,
            verbose=False,
            augment=False,
            max_det=300,
        )
        for p, res in zip(batch, results):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            image_id = int(p.stem)
            for (x1, y1, x2, y2), score in zip(xyxy, scores):
                dets.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dets))
    print(f"[eval] wrote {len(dets)} detections to {out_path} in {time.time() - t0:.1f}s")
    return len(dets)


def eval_coco(gt_path: Path, pred_path: Path) -> dict[str, float]:
    """Score COCO detections with pycocotools and return official-style metrics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(str(pred_path))
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {
        "mAP50-95": float(ev.stats[0]),
        "mAP50": float(ev.stats[1]),
        "mAP75": float(ev.stats[2]),
        "AP_small": float(ev.stats[3]),
        "AP_medium": float(ev.stats[4]),
        "AP_large": float(ev.stats[5]),
        "AR_100": float(ev.stats[8]),
        "AR_small": float(ev.stats[9]),
        "AR_medium": float(ev.stats[10]),
        "AR_large": float(ev.stats[11]),
    }


def maybe_start_wandb(args, config: dict):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "W&B requested but not installed. Install with: uv pip install wandb"
        ) from exc
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.name or None,
        tags=[t for t in args.wandb_tags.split(",") if t],
        config=config,
    )


def wandb_artifact(wandb_run, name: str, artifact_type: str):
    if not wandb_run:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "W&B run is active but wandb import failed. Install with: uv pip install wandb"
        ) from exc
    return wandb.Artifact(name, type=artifact_type)


def apply_nwd_loss_patch(alpha: float, c: float) -> None:
    patch_path = Path(__file__).resolve().parent / "nwd_loss_patch.py"
    spec = importlib.util.spec_from_file_location("clearsar_nwd_loss_patch", patch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load NWD patch from {patch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_nwd_patch(alpha=alpha, c=c)


def apply_qfl_loss_patch(gamma: float, alpha: float) -> None:
    patch_path = Path(__file__).resolve().parent / "qfl_loss_patch.py"
    spec = importlib.util.spec_from_file_location("clearsar_qfl_loss_patch", patch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load QFL patch from {patch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_qfl_patch(gamma=gamma, alpha=alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional weights to transfer into a YAML architecture, e.g. --model yolo26l-p2.yaml --pretrained yolo26l.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--cache", default=False, help="True/ram/disk for dataset caching")
    parser.add_argument("--project", default="runs/yolo",
                        help="Ultralytics output project directory.")
    parser.add_argument("--device", default=None,
                        help="Override training device. Default auto: cuda/mps/cpu.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of training data to use. Useful for local CPU smoke tests.")
    parser.add_argument("--name", default="", help="Custom run name suffix")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--freeze", type=int, default=None, help="Number of layers to freeze")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer to use (AdamW, MuSGD, etc.)")
    parser.add_argument("--data", default=None, help="Override dataset yaml path (bypasses fold-based lookup)")
    parser.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--cos-lr", action="store_true", help="Cosine LR schedule (better than linear for small datasets)")
    parser.add_argument("--rect", action="store_true",
                        help="Rectangular training: batches images by aspect ratio, letterbox to (imgsz x imgsz). "
                             "Requires imgsz scalar — YOLO rect=True with scalar imgsz is stable; "
                             "tuple form (H,W) triggers ultralytics issue #21730.")
    parser.add_argument("--no-val", action="store_true",
                        help="Disable Ultralytics validation during training. Use only for smoke tests.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable training plots. Useful on disk-constrained local smoke tests.")
    parser.add_argument("--experiment", default="ClearSAR-YOLO")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow logging. Useful for disk-constrained local runs.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="clearsar-phase2")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags")
    parser.add_argument("--eval-val-dir", default=None, help="Optional val image dir for post-train COCO eval")
    parser.add_argument("--eval-gt", default=None, help="Optional COCO GT json for post-train eval")
    parser.add_argument("--eval-cat-id", type=int, default=0, help="category_id for val detections")
    parser.add_argument("--copy-paste-rfi", type=float, default=0.0,
                        help="Ultralytics native copy_paste prob. For box-only ClearSAR labels this is usually a no-op; prefer external BoxPaste datasets.")
    parser.add_argument("--translate", type=float, default=0.05,
                        help="YOLO translate augmentation. Keep conservative for tiny SAR boxes.")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="YOLO scale augmentation. Keep conservative so 5-20 px boxes are not over-shrunk.")
    parser.add_argument("--box", type=float, default=12.0,
                        help="YOLO box loss weight. RFI boxes are thin; proven pseudo recipe used 12.0.")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="YOLO mixup probability. Pseudo-FT recipe uses 0.0; legacy V5.5 used 0.1.")
    parser.add_argument("--nwd-loss", action="store_true",
                        help="Apply Normalized Wasserstein Distance loss patch (small-target focus)")
    parser.add_argument("--nwd-alpha", type=float, default=0.5,
                        help="Weight on pixel-space NWD in blended box loss (0=pure CIoU, 1=pure NWD)")
    parser.add_argument("--nwd-c", type=float, default=12.8,
                        help="Pixel-space NWD normalization constant")
    parser.add_argument("--qfl-loss", action="store_true",
                        help="Opt-in Varifocal/quality-focal class loss using assigner target_scores")
    parser.add_argument("--qfl-gamma", type=float, default=2.0)
    parser.add_argument("--qfl-alpha", type=float, default=0.75)
    parser.add_argument(
        "--v12-baseline",
        type=float,
        default=0.51646,
        help="Baseline mAP for delta logging. Historical name; can be V12 or the honest F0 meta_val baseline.",
    )
    args = parser.parse_args()

    # os.chdir(Path(__file__).parent.parent)
    device = args.device or get_device()
    print(f"Device: {device}")

    # Dataset resolution: explicit --data > CLAHE fold > standard fold
    if args.data:
        dataset_yaml = Path(args.data)
    else:
        dataset_clahe = Path(f"data/yolo_clahe/fold{args.fold}/dataset.yaml")
        dataset_std = Path(f"data/yolo/fold{args.fold}/dataset.yaml")
        dataset_yaml = dataset_clahe if dataset_clahe.exists() else dataset_std

    assert dataset_yaml.exists(), f"Missing dataset: {dataset_yaml}"

    model_tag = Path(args.model).stem  # yolo26x.pt -> yolo26x
    run_name = f"{model_tag}_fold{args.fold}_sz{args.imgsz}{'_rect' if args.rect else ''}_ep{args.epochs}_{args.optimizer}"
    save_dir = Path(args.project).resolve()

    run_config = {
        "model": args.model,
        "pretrained": args.pretrained,
        "fold": args.fold,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "cache": args.cache,
        "workers": args.workers,
        "fraction": args.fraction,
        "optimizer": args.optimizer,
        "device": device,
        "rect": args.rect,
        "val": not args.no_val,
        "plots": not args.no_plots,
        "cos_lr": args.cos_lr,
        "freeze": args.freeze,
        "lr0": args.lr0,
        "dataset": str(dataset_yaml),
        "project": str(save_dir),
        "v12_baseline": args.v12_baseline,
        "copy_paste_rfi": args.copy_paste_rfi,
        "translate": args.translate,
        "scale": args.scale,
        "box": args.box,
        "mixup": args.mixup,
        "nwd_loss": args.nwd_loss,
        "nwd_alpha": args.nwd_alpha,
        "nwd_c": args.nwd_c,
        "qfl_loss": args.qfl_loss,
        "qfl_gamma": args.qfl_gamma,
        "qfl_alpha": args.qfl_alpha,
    }

    wandb_run = maybe_start_wandb(args, run_config)

    if args.no_mlflow:
        print("MLflow disabled")
        try:
            from ultralytics.utils import SETTINGS

            SETTINGS["mlflow"] = False
        except Exception:
            pass
        mlflow_ctx = contextlib.nullcontext()
    else:
        # MLflow: default to local file store. Override via env if you want HTTP server.
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"file://{Path('mlruns').resolve()}")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI: {tracking_uri}")
        mlflow.set_experiment(args.experiment)
        mlflow_ctx = mlflow.start_run(run_name=run_name)

    with mlflow_ctx:
        if not args.no_mlflow:
            mlflow.log_params(run_config)

        if args.nwd_loss:
            apply_nwd_loss_patch(alpha=args.nwd_alpha, c=args.nwd_c)
            print(f"[nwd] pixel-space NWD loss patch applied alpha={args.nwd_alpha} C={args.nwd_c}", flush=True)
        if args.qfl_loss:
            apply_qfl_loss_patch(gamma=args.qfl_gamma, alpha=args.qfl_alpha)
            print(f"[qfl] varifocal class-loss patch applied gamma={args.qfl_gamma} alpha={args.qfl_alpha}", flush=True)

        # One-line recipe manifest — makes silent default shifts (optimizer/box/mixup/
        # translate/scale) visible in any log scrape. Defaults moved to pseudo-FT recipe;
        # legacy V5.5 callers must pin these explicitly.
        print(
            f"[recipe] optimizer={args.optimizer} lr0={args.lr0} cos_lr={args.cos_lr} "
            f"freeze={args.freeze} box={args.box} mixup={args.mixup} "
            f"translate={args.translate} scale={args.scale} "
            f"copy_paste_rfi={args.copy_paste_rfi} nwd={args.nwd_loss} "
            f"(alpha={args.nwd_alpha} C={args.nwd_c}) qfl={args.qfl_loss} "
            f"(gamma={args.qfl_gamma} alpha={args.qfl_alpha})",
            flush=True,
        )

        model = YOLO(args.model)
        if args.pretrained:
            print(f"Loading pretrained transfer weights: {args.pretrained}", flush=True)
            model.load(args.pretrained)
        results = model.train(
            data=str(dataset_yaml.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            cache=args.cache,
            device=device,
            patience=args.patience,
            project=str(save_dir),
            name=args.name if args.name else f"fold{args.fold}",
            rect=args.rect,
            freeze=args.freeze,
            # Augmentation: geometric only, no HSV (SAR calibrated)
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,      # SAR geometry is axis-aligned
            shear=0.0,
            perspective=0.0,
            flipud=0.0,       # NO vertical flip: RFI is azimuth-aligned, flip changes physics
            fliplr=0.0,       # V5.5 update: fliplr=0.0 (lethal to mAP)
            mosaic=1.0,       # mosaic for small dataset
            mixup=args.mixup,
            copy_paste=args.copy_paste_rfi,
            translate=args.translate,
            scale=args.scale,
            # Thin-object localization: upweight bbox regression loss.
            # RFI boxes are 5-20px tall; 1px vertical error fails AP75.
            box=args.box,
            # Objectness / NMS tuning for thin objects
            overlap_mask=False,
            close_mosaic=10,
            # Save
            save=True,
            save_period=20,
            val=not args.no_val,
            plots=not args.no_plots,
            workers=args.workers,
            seed=42 + args.fold,
            fraction=args.fraction,
            # Optimizer
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=0.01,
            cos_lr=args.cos_lr,
            weight_decay=5e-4,
            warmup_epochs=3,
            amp=True,
        )

        # Log best mAP metrics
        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        for k, v in metrics.items():
            try:
                key = k.replace("(", "").replace(")", "").replace("/", "_")
                if not args.no_mlflow:
                    mlflow.log_metric(key, float(v))
                if wandb_run:
                    wandb_run.log({f"ultralytics/{key}": float(v)})
            except Exception:
                pass

        best_map = metrics.get("metrics/mAP50-95(B)", 0)
        print(f"\nFold {args.fold} best mAP50-95: {best_map:.4f}")
        if not args.no_mlflow:
            mlflow.log_metric("best_mAP50_95", best_map)
        if wandb_run:
            wandb_run.log({"ultralytics/best_mAP50_95": best_map})

        run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else save_dir / (args.name if args.name else f"fold{args.fold}")
        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists() and not args.no_mlflow:
            mlflow.log_artifact(str(best_weights))
            if wandb_run:
                artifact = wandb_artifact(wandb_run, f"{args.name or run_name}-best", "model")
                artifact.add_file(str(best_weights))
                wandb_run.log_artifact(artifact)

        if args.eval_val_dir and args.eval_gt:
            if not best_weights.exists():
                last_weights = run_dir / "weights" / "last.pt"
                if not last_weights.exists():
                    raise FileNotFoundError(
                        f"Neither best nor last weights found for eval: {best_weights}, {last_weights}"
                    )
                print(f"[eval] best.pt not found; using last.pt from no-val run: {last_weights}", flush=True)
                best_weights = last_weights
            eval_dir = run_dir / "coco_eval"
            pred_path = eval_dir / "val_preds.json"
            stats_path = eval_dir / "val_metrics.json"
            n_dets = predict_coco(
                best_weights,
                Path(args.eval_val_dir),
                pred_path,
                args.imgsz,
                args.eval_cat_id,
            )
            stats = eval_coco(Path(args.eval_gt), pred_path)
            stats["detections"] = float(n_dets)
            stats["delta_vs_baseline"] = stats["mAP50-95"] - args.v12_baseline
            # Backward-compatible key for old dashboards/scripts that consumed this name.
            stats["delta_vs_v12"] = stats["delta_vs_baseline"]
            stats_path.write_text(json.dumps(stats, indent=2))
            if not args.no_mlflow:
                for key, val in stats.items():
                    mlflow.log_metric(f"coco_{key}", float(val))
                mlflow.log_artifact(str(pred_path))
                mlflow.log_artifact(str(stats_path))
            print("\nCOCO eval:")
            for key, val in stats.items():
                print(f"  {key}: {val:.6f}")
            if wandb_run:
                wandb_run.log({f"coco/{key}": float(val) for key, val in stats.items()})
                artifact = wandb_artifact(wandb_run, f"{args.name or run_name}-val-preds", "predictions")
                artifact.add_file(str(pred_path))
                artifact.add_file(str(stats_path))
                wandb_run.log_artifact(artifact)

    print(f"Run saved to {save_dir}/fold{args.fold}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
