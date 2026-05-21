"""
Normalized Wasserstein Distance (NWD) loss patch for ultralytics' BboxLoss.

Reference: Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny
Object Detection" (AI-TOD); follow-up in AI-TOD-v2. The intuition: model each
bbox as a 2D Gaussian (center = mean, w/2 and h/2 = std), compute the L2
Wasserstein distance between predicted and target Gaussians, and use
exp(-sqrt(W2) / C) as a similarity. Unlike IoU, NWD is smooth for non-
overlapping tiny boxes and gives a usable gradient.

For RFI detection (boxes 5-20 px on a 640 px image) we follow the AI-TOD
recipe: blend NWD with CIoU at alpha=0.5. Ultralytics passes boxes into
BboxLoss in stride-relative grid units, so this patch converts the positive
boxes back to pixel coordinates before computing NWD. CIoU stays in the native
grid units because it is scale-invariant.

Apply once before model.train():
    from nwd_loss_patch import apply_nwd_patch
    apply_nwd_patch(alpha=0.5, c=12.8)
"""
from __future__ import annotations

_PATCHED = False


def apply_nwd_patch(alpha: float = 0.5, c: float = 12.8) -> None:
    """Monkey-patch ultralytics.utils.loss.BboxLoss.forward to blend NWD with CIoU.

    alpha: weight on NWD term (0 = pure CIoU, 1 = pure NWD). 0.5 is the
        AI-TOD recommendation and reasonable for RFI.
    c: NWD normalization constant in pixels. 12.8 is the AI-TOD default and
        appropriate for tiny boxes in the ~5-20 px range.
    """
    global _PATCHED
    if _PATCHED:
        return

    import torch
    import torch.nn.functional as F
    from ultralytics.utils import loss as _loss
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist

    BboxLoss = _loss.BboxLoss
    original_forward = BboxLoss.forward

    def _nwd(pred_xyxy: torch.Tensor, targ_xyxy: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        pcx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) * 0.5
        pcy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) * 0.5
        tcx = (targ_xyxy[:, 0] + targ_xyxy[:, 2]) * 0.5
        tcy = (targ_xyxy[:, 1] + targ_xyxy[:, 3]) * 0.5
        pw = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0)
        ph = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
        tw = (targ_xyxy[:, 2] - targ_xyxy[:, 0]).clamp(min=0)
        th = (targ_xyxy[:, 3] - targ_xyxy[:, 1]).clamp(min=0)
        center = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        wh = ((pw - tw) ** 2 + (ph - th) ** 2) * 0.25
        w2 = center + wh
        return torch.exp(-torch.sqrt(w2 + eps) / c)

    def patched_forward(self,
                        pred_dist, pred_bboxes, anchor_points, target_bboxes,
                        target_scores, target_scores_sum, fg_mask,
                        imgsz, stride):
        """Match ultralytics 8.4.47 BboxLoss.forward signature, blend CIoU+NWD."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred = pred_bboxes[fg_mask]
        targ = target_bboxes[fg_mask]
        iou = bbox_iou(pred, targ, xywh=False, CIoU=True)
        # pred/targ are in normalized [0,1] image coords (divided by imgsz).
        # Scale to pixel space for NWD using imgsz directly — avoids MPS
        # stride-tensor indexing bug (index out of bounds on expand+bool-index).
        scale = float(imgsz[0])  # assume square; imgsz=[H,W]
        nwd_sim = _nwd(pred * scale, targ * scale)
        if nwd_sim.dim() < iou.dim():
            nwd_sim = nwd_sim.unsqueeze(-1)
        blend = (1.0 - alpha) * iou + alpha * nwd_sim
        loss_iou = ((1.0 - blend) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                     target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_sc = pred_dist * stride
            pred_dist_sc[..., 0::2] /= imgsz[1]
            pred_dist_sc[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist_sc[fg_mask], target_ltrb[fg_mask], reduction="none")
                .mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    BboxLoss.forward = patched_forward
    BboxLoss._nwd_alpha = alpha
    BboxLoss._nwd_c = c
    BboxLoss._original_forward = original_forward
    _PATCHED = True
