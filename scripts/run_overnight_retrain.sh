#!/usr/bin/env bash
# Overnight retrain: cross-validate the small datasets (no official split exists for
# any of them, so we use our own stratified group k-fold), then finish BUS-BRA's
# remaining official-fold runs (fold 1 already done in runs/run-1). Everything uses
# the --paper-match recipe (SGD momentum 0.9 lr 1e-3, weighted CE, 100 epochs no
# early stopping, contrast stretching where supported) for a consistent, rigorous
# training budget across datasets, since none of these smaller datasets publish
# their own full training recipe the way BUS-BRA does.
set -e
cd /home/nouran/MMFM
mkdir -p runs/run-2
PY=.venv/bin/python3

run() {
  local dataset=$1 image_size=$2 wandb_project=$3
  echo "=== $(date) starting $dataset 5-fold ==="
  $PY training/train.py --dataset "$dataset" --single-mode --kfold 5 \
    --epochs 100 --batch-size 32 --image-size "$image_size" --backbone resnet18 --pretrained \
    --paper-match --device cuda --wandb-project "$wandb_project" \
    --wandb-run-name "${dataset}-paper-match-5fold" \
    --checkpoint-path "runs/run-2/${dataset}_single_resnet18_best.pth" \
    > "runs/run-2/${dataset}_5fold_run.log" 2>&1
  echo "=== $(date) finished $dataset 5-fold ==="
}

# fastest/smallest first so as much as possible is done by morning
run mias 224 mmfm-mias
run busc 128 mmfm-busc
run breast 224 mmfm-breast
run busi 224 mmfm-busi

# finish BUS-BRA's remaining official folds (fold 1 already in runs/run-1). Each
# single-fold call writes the *same* busbra_5fold_resnet18_results.json filename in
# cwd, so it must be renamed immediately after each fold or the next fold overwrites it.
for fold in 2 3 4 5; do
  echo "=== $(date) starting busbra fold $fold ==="
  $PY training/train.py --dataset busbra --single-mode --busbra-kfold 5 --busbra-test-fold "$fold" \
    --epochs 100 --batch-size 32 --image-size 224 --backbone resnet18 --pretrained \
    --paper-match --device cuda --wandb-project mmfm-busbra \
    --wandb-run-name "busbra-paper-match-fold${fold}" \
    --checkpoint-path "runs/run-2/busbra_single_resnet18_best.pth" \
    > "runs/run-2/busbra_fold${fold}_run.log" 2>&1
  mv -f ./busbra_5fold_resnet18_results.json "runs/run-2/busbra_fold${fold}_results.json" 2>/dev/null || true
  echo "=== $(date) finished busbra fold $fold ==="
done

# results-summary JSONs for the small-dataset 5-fold runs land in the repo root
mv -f ./mias_5fold_resnet18_results.json ./busc_5fold_resnet18_results.json \
      ./breast_5fold_resnet18_results.json ./busi_5fold_resnet18_results.json \
      runs/run-2/ 2>/dev/null || true

echo "=== $(date) ALL DONE ==="
