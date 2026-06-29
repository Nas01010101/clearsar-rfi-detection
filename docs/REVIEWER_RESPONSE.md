# Response to reviewers — IEEE ICIP 2026, abstract #4319

*RFI Detection in Sentinel-1 SAR Imagery via Pseudo-Label Distillation and Multi-Resolution Ensemble*

We thank the reviewers and chairs. Both substantive points concern the
**honesty of the ablation on genuinely unseen data**, and we have
restructured the experiments accordingly. Below, each comment is quoted
and answered with the concrete change made to the camera-ready.

---

## Reviewer TP-1018

> The data-driven derivation of design constraints from RFI stripe geometry, the leak-isolated meta-validation, the quantified negative results, and the explicit acknowledgement of teacher leakage are welcome, especially in a challenge paper.
>
> One thing to clarify. On genuinely unseen data, the entire student distillation, multiresolution TTA, fold dropping, and blending stack add very little compared to simply submitting the teacher ensemble. It would be great to report each component's contribution on the held-out test set, or on a split where V12 is not leaked, so the ablation reflects what can be gained from using the pipeline on unseen data.

**This is correct, and we now report it head-on as the paper's central
finding rather than burying it.** We took the reviewer's *first* offered
option — reporting on the genuine held-out test set — because it is the
strongest possible measure of "what can be gained on unseen data" (the
real 786-image ESA partition at full pipeline scale), and because the
relevant end-points were already scored by the challenge platform.

Changes:

1. **New held-out-test ablation table** (Table 1 in the report;
   Table 1(b) in the abstract), scored by the ESA platform on the
   undisclosed test partition:

   | Configuration | mAP@[.50:.95] | Δ |
   |---|---|---|
   | Plain 5-fold WBF ensemble | 0.4720 | — |
   | Teacher V12 (+RF-DETR, CLAHE swap) | 0.4755 | +0.0035 |
   | Full pipeline V17 | **0.4776** | +0.0021 |

   So on unseen data the **entire** distillation + multi-resolution TTA +
   fold-dropping + blending stack adds only **+0.0021** over submitting
   the teacher, and **+0.0056** over a plain 5-fold ensemble. We also
   note explicitly that the stack does *not extend*: raising the blend
   weight to w=0.375 regressed to 0.4768, and a more aggressive
   second-generation pseudo-FT blend regressed to 0.4566 — **below** the
   teacher.

2. **The previous (meta-validation) ablation is relabelled** as what it
   actually measures: the construction of the honest student source *in
   isolation* (Table 2 / Table 1(a)). Its +0.077 gain is real but, on
   unseen test, **largely redundant with the teacher**, which already
   encodes the same multi-fold knowledge. We state this redundancy as
   the explicit lesson connecting the two tables.

3. **Why a full per-component chain on held-out test is not shown:**
   official submissions were rate-limited to one per 12 h, so only the
   three ensemble end-points above were scored on the true test
   partition; the intermediate distillation stages were measured on the
   leak-isolated meta-validation set. We say this plainly rather than
   implying the meta-val deltas transfer.

4. **Discussion rewritten** to give the operationally honest
   recommendation: a quality-weighted multi-fold ensemble teacher
   recovers most of the attainable accuracy on fresh SAR data; the
   distillation/TTA/blend stack is a small, fragile top-up worthwhile
   only at a conservative blend weight and at substantial extra compute.

The meta-validation chain is now reproducible from the archived
per-fold/per-resolution predictions via `scripts/verify_ablation.py`
(independently re-confirmed: 0.434 → 0.469 → 0.471).

---

## Reviewer SS-2426

> Some details in the manuscript can be improved and better described. It is also unclear how the cross-validation was implemented as it seems oddly is part of the methods, rather than used for evaluation (as it is conventionally done).

**Clarified.** The reviewer correctly noticed that the five-fold split is
*not* used in the conventional way (as an out-of-fold generalisation
estimator). We added an explicit **"Role of cross-validation"**
paragraph to the evaluation protocol (and a compact version to the
abstract's Task-and-Data section) stating that:

- the five-fold split serves an **ensemble-construction** role — each
  fold detector becomes a member of the WBF teacher ensemble;
- cross-fold **validation** mAPs are leaked (every fold detector has,
  through the other four folds, seen the images held out by any single
  fold) and are therefore **never** used to estimate test performance or
  to select the final model;
- honest generalisation is measured by **two separate instruments**: the
  401-image leak-isolated meta-validation set (a single-model gate only)
  and the official held-out test set (all model selection and final
  reporting).

This removes the ambiguity about where cross-validation sits between
"method" and "evaluation": it is a method device for building ensemble
diversity, and evaluation is handled by the two dedicated instruments
above.

---

## Additional camera-ready improvements

Beyond the two reviewer points, this revision also:

- **Size-/IoU-stratified ablation.** Table 2 now reports AP / AP$_{50}$ /
  AP$_{75}$ / AP$_S$ (small-area), as expected for tiny-object detection.
  AP$_S$ rises $0.356\to0.418$ along the chain; AP$_{75}$ is the weakest
  column ($0.522$ vs $0.752$ at AP$_{50}$), locating the residual error
  at high IoU. All numbers reproduced by `scripts/verify_ablation.py`.
- **Three-item contributions list** added at the end of the
  introduction (field convention for detection papers).
- **Citation audit.** All 19 references verified against primary
  sources (real, correctly attributed; arXiv IDs / venues / pages
  checked). YOLO26 and its named modules (STAL, ProgLoss, MuSGD, P2
  head) confirmed real.
- **Factual corrections (verified against the archived training
  configs).** Base-detector parameter count corrected from
  $\approx$55.7\,M to $\approx$59\,M (measured directly from the
  released `yolo26x.pt`: 58{,}993{,}368 parameters). Base-training
  hyperparameters corrected to the actual run (`phase_a_fold0-2/args.yaml`,
  ultralytics 8.4.47): batch 16$\to$32, patience 20$\to$15, epoch cap
  120$\to$60, and the learning-rate schedule cosine$\to$linear (final
  $\eta_f{=}10^{-5}$ unchanged). The student fine-tune (15 epochs) was
  confirmed directly from the Thunder cloud run logs
  (`runs\_thunder\_v18a/fold\{1,2,4\}/results.csv`). MuSGD, box-gain 12.0,
  NWD ($\alpha{=}0.5$, $C{=}12.8$), 3-epoch warmup, and 640\,px were
  confirmed correct, as were
  all dataset statistics (median image $515\times342$, box $140\times10$\,px,
  aspect $\approx$8:1) and the CLAHE settings (clip 3.0, $8\times8$, LAB-L).
- **Number consistency.** The TTA delta is $+0.035$ (not $+0.036$), so
  the meta-validation deltas sum to the reported $+0.077$.
- **Tone pass.** Removed editorialising/LLM-tell phrasing throughout,
  matching the terse, factual register of strong ICIP detection papers.

## Files changed

- `docs/clearsar_paper.tex` — 5-page report (new held-out Table 1;
  size-stratified Table 2; contributions list; eval-protocol and
  discussion rewritten; param count corrected; tone pass).
- `docs/clearsar_abstract_2p.tex` — 2-page camera-ready abstract (same
  changes, compacted; combined ablation table with panels (a)/(b);
  abstract trimmed to 142 words).
- `scripts/verify_ablation.py` — reproduces the meta-val ablation chain
  with the full AP / AP$_{50}$ / AP$_{75}$ / AP$_S$ breakdown.
- `README.md` — results section updated to the two-table framing.
