# To-do: paper and TIK2 progress report

Source document: `~/Desktop/paper/main.tex` (report "Towards a Modality-Agnostic Breast
Cancer Classifier"). Plan of record: `~/Desktop/NF_DoktoraTezÖneri.pdf` (thesis proposal,
17/06/2026, Aims 1–2, four-term Gantt). Started 2026-11-11.

Status markers: `[ ]` open · `[~]` in progress / partial · `[x]` done

---

## A. Paper (first target: data audit + reproduction paper)

### A1. Reviewer weak points (in progress)
- [~] **Leakage-corrected baselines.** Re-run single-modality CV with duplicate groups kept
      inside one fold (`--duplicate-groups`, uncommitted in `training/train.py`).
  - [x] BUSC, ResNet-18, row vs. dup split, seeds 42–44 (`runs/busc_dup/`, 2026-10-29).
        Accuracy 0.999 (row) → ~0.981 (dup): leakage does **not** explain BUSC's score.
  - [~] BUSC: stronger grouping by consecutive file number. Files are numbered by class
        (benign us1–100, malignant us101–250) and all 41 audited duplicate pairs are 1–3
        numbers apart, so frames are in acquisition order. New `--sequence-block-groups K`
        (blocks of 10 → 25 groups); script `runs/busc_seq/run.sh`, 3 seeds.
  - [~] BUSI, ResNet-18, row vs. dup, 3 seeds — script `runs/busi_dup/run.sh`
        (original BUSI 5-fold took ~6 h; 6 runs ≈ up to 36 h, check speed first).
  - [ ] BUSI, EfficientNet-B0, dup split.
  - [ ] CMMD: merge duplicated patient IDs (D1-0202/D2-0284, D1-0014/D1-0269,
        D1-0808/D1-1292); drop or adjudicate the conflicting pair.
  - [x] BUS-UCLM CHCO/CHVI already merged in `_bus_uclm_metadata`; DMID duplicates dropped
        via `DMID_DUPLICATES`.
  - [ ] Update report tables (BUSI, BUSC) and the audit's "recommendation 1" status.
- [ ] **One evaluation protocol.** Write a protocol table (optimizer, epochs, selection rule,
      split, seeds) and bring every reported number under it, or label exceptions.
  - [ ] Held-out *test* split for the fusion runs (currently validation-only, and
        checkpoints are selected on that same validation set). Proposal: for fold k, the
        fusion model trains/validates/tests on each dataset's own fold-k train/val/test
        frames, so fusion and specialists are scored on identical test images.
  - [x] **Warm-start leakage — CONFIRMED (2026-11-11).** The Section 13 (eleven-source) fusion
        runs warm-started the US branch from `bus_cot_single_resnet18_best.pth`, trained on
        BUS-Lesion trainval, which contains 824 BUS-BRA and 439 BUSI lesion crops. Their
        BUSI/BUS-BRA validation AUCs (0.91–1.00) are contaminated → re-run under the
        fold-aligned protocol; US warm-start must come from a fold-k model without those copies.
  - [~] **Fold-aligned fusion protocol implemented** (`--combined-fold K`, uncommitted):
        each source uses its single-mode k-fold split (fixed official split for CBIS-DDSM,
        BreastDM, ODELIA, AMBL; ISPY2 train/val only), best checkpoint scored per source on
        test → `*_test_source_metrics.json` + per-sample `.npz`; `{fold}` in init-checkpoint
        paths. Single-mode k-fold runs now also save `*_test_predictions.npz`.
        Still to do: smoke test; fusion-vs-specialist comparison script (paired DeLong/bootstrap).
  - [ ] Fusion's BUSI/BUSC sources use plain random splits (`_busi_frame`) — same duplicate
        leakage as the baselines; switch to grouped splits.
  - [ ] Selection rule: validation loss picked poor checkpoints for ODELIA, CBIS-DDSM and
        the fusion runs; standardise on validation AUC (document per section).
  - [ ] ≥3 seeds for every headline number; report mean ± std across seeds and folds.
  - [ ] Mixed-backbone table: Baseline ran 30 epochs vs. stated 10-epoch budget; fix or re-run.
- [ ] **Shortcut tests that the report proposes but never ran.**
  - [ ] BUSC framing test: crop/mask the scan border or recover a lesion ROI, re-run.
  - [ ] mini-MIAS / CMMD / CDD-CESM border attention: tissue-masked or lesion-cropped re-run.
  - [ ] BUS-BRA / BUSI calipers: inpaint markers (caliper-cross removal), re-run; compare
        per-image predictions on annotated vs. clean images.
  - [ ] ISPY2 vs. BreastDM: same crop rule + fixed physical size; re-run source probe and
        fusion MRI branch; report per-source MRI metrics.
  - [ ] DMID: laterality-flip (all images to one side) and laterality-only baseline.

- [~] **FusionLateModel BatchNorm contamination — FIXED in code (2026-11-11, uncommitted).**
      `forward` ran every backbone on every sample, so in combined training ~2/3 of each
      branch's batch were zero-filled placeholders that entered BatchNorm statistics (and
      overwrote the warm-started branches' BN stats). Likely mechanism behind "joint training
      erodes the warm-started branches" and "frozen backbones work best" (Section 13). Now each
      backbone runs only on present samples (~3× less compute); eval outputs of old checkpoints
      unchanged. Regression tests in `tests/test_fusion_model.py` (fail on old code).
      To do: re-run the Section 13 lr/frozen comparison to test the explanation; update text.

- [~] **GPU queue running** (`runs/fold_aligned/queue.sh`, log `runs/fold_aligned/queue.log`,
      started 2026-11-11 16:40, detached): BUSC seq-block CV → fold-aligned fusion 5 folds ×
      {all trained, frozen} → BUSI row vs dup (3 seeds) → specialists (mias, breast, busc-dup,
      dmid, bus_uclm, bcsdbt_full, bus_cot; breamdm, odelia, ambl, cbis_ddsm).
      BUS-BRA specialist predictions written from existing fold checkpoints
      (`scripts/predict_busbra_folds.py`; AUCs match the recorded ones exactly).
      Then: `scripts/compare_fusion_specialists.py` → fusion-vs-specialist table (Obj. 1.2).
- [ ] Note for the report: the Oct-29 BUSC row/dup runs (`runs/busc_dup`) trained **from
      scratch** (`--paper-match` default recipe sets no `pretrained`) and still reached
      0.98–1.0; the reported baseline and all new runs use ImageNet weights.

- [~] **MICCAI first draft** `docs/miccai_draft/main.tex` (2026-11-11): leakage-and-shortcuts
      story, 8 pp incl. references (content ≈5.5 pp). Fallback class — re-typeset in real
      `llncs` + `splncs04` before submission. Red `\pending{}` cells: BUSC/BUSI dup-grouped
      rows, frozen fusion column, specialist column. Room left for a protocol figure.

## >>> NEXT SESSION START HERE (written 2026-11-12 21:45) <<<
GPU queue finished 2026-11-12 15:51 (all steps). Nothing running. Results on disk:
- BUSC frame blocks (runs/busc_seq): acc 0.748/0.824/0.788, AUC 0.949/0.930/0.925 (seeds 42-44)
  vs random split 0.992 → leakage between consecutive frames. Specialist busc (dup groups,
  pretrained): runs/specialists/busc_results.json.
- BUSI (runs/busi_dup): row AUC 0.939/0.937/0.931, dup 0.945/0.941/+s44 → duplicates don't matter.
- Fusion fold-aligned (runs/fold_aligned): all-trained test mean-branch AUC 0.796±0.008,
  frozen 0.779±0.010; all-trained wins every fold; frozen better on MRI (AMBL 0.716 vs 0.576).
- Fusion vs specialist (runs/fold_aligned/fusion_{all,frozen}_vs_specialists.json), all-trained:
  fusion better on mini-MIAS (+0.12), BrEaST (+0.06), BUS-UCLM (+0.17); worse on DMID (−0.07),
  BUS-BRA (−0.02), BreastDM (−0.04); rest n.s.
TODO next, in order:
1. [x] (2026-11-17, per-fold paired deltas + bootstrap) **Fix the comparison stats for fixed-split sources** (CBIS-DDSM, BreastDM, ODELIA, AMBL):
   their test set is identical in all 5 folds, so pooling folds counts each image 5×
   (n=2230 for BreastDM = 446×5) → CI/p too optimistic. Average the 5 fusion probabilities
   per image (or compare per fold) before DeLong/bootstrap. K-fold sources are fine.
2. [x] (footnoted in both tables) Note: MRI + CBIS specialists used the default Adam recipe with early stopping (7-10 epochs),
   not --paper-match; say so in the table or re-run with a stronger recipe.
3. [~] (running: runs/fold_aligned/busc_seq_fusion.sh, started 2026-11-17 16:21) Re-run fusion with BUSC grouped by frame blocks (`--sequence-block-groups 10`), all-trained,
   5 folds (~45 min/fold); BUSC is still leaky in the current fusion runs (0.998).
4. [~] (all filled except BUSC fusion footnote) Fill the MICCAI draft's red cells (docs/miccai_draft): BUSC/BUSI dup rows, frozen column,
   specialist column (after step 1); add the frozen/AMBL observation.
5. [x] (2026-11-17: abstract, new 4.2.4 leak-controlled, Sec. 5 update, new 6.5 fold-aligned) Update the TIK2 report (~/Desktop/paper): abstract (freezing claim is reversed), Sec. 5
   BUSC discussion (leakage, not framing), Sec. 6 with the fold-aligned results.
6. Upload new weights to the private HF repo noe95/mmfm-checkpoints: fusion_all_fold{1-5},
   busi_dup_s42 folds, busc_seq10_s42 folds; update docs/hf_checkpoints_card.md.
7. Commit (code/docs only): scripts/compare_fusion_specialists.py fix, docs/miccai_draft,
   docs/hf_checkpoints_card.md, todo, runs/* JSON/NPZ. ~/Desktop/paper is outside the repo.
8. Disk: ~3.5 GB free; fusion checkpoints in runs/fold_aligned (~1.4 GB) can go once on HF.

### A2. Content still missing for the paper
- [ ] Fusion vs. single-dataset specialist, per source, same test split (Objective 1.2).
- [ ] Leave-one-source-out / leave-one-country-out evaluation (Objective 2.2).
- [ ] Literature positioning: `siddique2026audit`; prior BUSI duplicate/quality critiques;
      CBIS-DDSM mass/calc split issue; check which audit findings are new.
- [~] Venue: MICCAI-style draft started; confirm year's page limit and anonymity rules.
- [ ] Remove repo paths / script names from the prose (old "point 5").

### A3. Small fixes
- [ ] `\totalUsed` macro (53,166) ≠ sum of "used" column (53,141); fix in
      `scripts/dataset_report_assets.py`.
- [ ] Integrity table: BUS-CoT markers = text = 11,439 (= every image) — detector failure?
- [ ] `busuclm2025mendeley` bib entry has no author (BibTeX sort warning).

---

## B. TIK2 progress report (covers Term 1 **and** Term 2 of the proposal)

### B1. Term 1 — data curation, preprocessing, baseline evaluation
- [x] Data intake: 16 public datasets, three modalities (audit, Section 3).
- [ ] Reconcile with the proposal's dataset list: INbreast, BCDR, OASBUD, Rodtook,
      Duke-Breast-MRI, MAMA-MIA not included; UDIAT (Dataset B) is inside BUS-CoT;
      proposal's BUS-BRA count (5,625) vs. 1,875 on disk. State reasons.
- [~] Statistical vs. generative artifact removal: line removal measured (destructive),
      CycleGAN learns calipers. Missing: a quantitative comparison and a training run that
      uses any preprocessing step.
- [ ] CFE-GAN (USenhance-2023 variant) intensity harmonization: train, evaluate, report.

### B2. Term 2 — backbone selection, single-modality benchmarking
- [x] CNN baselines (ResNet-18/50, EfficientNet-B0) per dataset; reproductions vs. papers.
- [ ] Attention-based baselines: Swin / ViT single-modality runs (proposal Term 2).
- [~] Hardware optimization: gradient accumulation used (MST, MedGemma); document
      mixed precision / memory use on the RTX 3060 (proposal says 3070).
- [ ] Per-modality baseline summary table (AUC, sensitivity, specificity) as the
      reference for the fusion model.

### B3. Report structure
- [ ] Restructure into: aims → work done per term vs. Gantt → key results → deviations
      from plan (datasets, hardware, fusion started early) → plan for Term 3.
- [x] Merged (2026-11-12): fusion → Sec. 6, single-modality baselines + reproductions → Sec. 4, VLMs → Sec. 7;
      drop superseded Section 7 content; fold Section 13 into the preprocessing part.
- [ ] Term 3 plan: paired multi-modality data source (none of the 16 datasets has the same
      patient in US + MG + MRI); gated multimodal attention fusion; missing-modality ablations;
      demographic metadata audit for Objective 2.1.

---

## Env
- [~] `.venv` rebuilt on Python 3.14 (2026-11-11; old cp312 env deleted, package list in
      session scratchpad). Re-check bitsandbytes/MedGemma and open_clip paths after install.
