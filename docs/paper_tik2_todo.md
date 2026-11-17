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

## >>> STATUS (2026-11-17 late) <<<
Done: comparison stats (per-fold paired deltas + bootstrap for every source; pooled OOF
DeLong kept in JSON for reference); BUSC frame-block multi-source re-run
(runs/fold_aligned/fusion_seq_*: mean per-branch test AUC 0.806±0.015; BUSC 0.998→0.951);
MICCAI draft + TIK2 report updated (no pending cells); "fusion" reframed as multi-source
training within modality (one modality per sample → branch weights inert); leak-free
BUSI/BUSC folds + multisource folds on HF (noe95/mmfm-checkpoints, private);
superseded fusion checkpoints deleted.
Headline (multi-source vs single-dataset, same test images): better BUS-UCLM +0.17, AMBL +0.07;
trend mini-MIAS +0.12 (p=.08), BrEaST +0.05 (p=.06); worse BreastDM −0.02; 8 n.s.
Next candidates:
- [ ] Seeds: per-source AUCs move up to 0.09 between runs (AMBL 0.58 vs 0.66) → 3 seeds of the
      fold-aligned multi-source run (~4 h each) before final paper numbers.
- [ ] Real cross-modality fusion needs paired multi-modality patients (none in the 16 sets)
      + the proposal's gated attention fusion (Term 3).
- [ ] Shortcut tests (A1 third bullet) still not run.
- [ ] Re-typeset MICCAI draft in real llncs (needs llncs.cls download).

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
