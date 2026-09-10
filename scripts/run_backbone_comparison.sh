#!/usr/bin/env bash
# Second-backbone comparison, matching each paper's own reported comparison model
# where one exists (BUS-BRA: ResNet50, its Table 3) or the literature-suggested
# follow-up backbone from dataset_reproduction_plan.md where the source paper
# itself reports none (BUSI, BrEaST: EfficientNet-B0). All 5-fold CV, same
# --paper-match recipe as the ResNet-18 runs for a fair comparison.
set -e
cd /home/nouran/MMFM
mkdir -p runs/run-2/extra-backbones
PY=.venv/bin/python3

run_busbra() {
  echo "=== $(date) starting busbra resnet50 5-fold ==="
  $PY training/train.py --dataset busbra --single-mode --busbra-kfold 5 \
    --epochs 100 --batch-size 32 --image-size 224 --backbone resnet50 --pretrained \
    --paper-match --device cuda --wandb-project mmfm-busbra \
    --wandb-run-name busbra-paper-match-resnet50-5fold \
    --checkpoint-path "runs/run-2/extra-backbones/busbra_single_resnet50_best.pth" \
    > "runs/run-2/extra-backbones/busbra_resnet50_5fold_run.log" 2>&1
  echo "=== $(date) finished busbra resnet50 5-fold ==="
}

run_kfold() {
  local dataset=$1 image_size=$2 backbone=$3 wandb_project=$4
  echo "=== $(date) starting ${dataset} ${backbone} 5-fold ==="
  $PY training/train.py --dataset "$dataset" --single-mode --kfold 5 \
    --epochs 100 --batch-size 32 --image-size "$image_size" --backbone "$backbone" --pretrained \
    --paper-match --device cuda --wandb-project "$wandb_project" \
    --wandb-run-name "${dataset}-paper-match-${backbone}-5fold" \
    --checkpoint-path "runs/run-2/extra-backbones/${dataset}_single_${backbone}_best.pth" \
    > "runs/run-2/extra-backbones/${dataset}_${backbone}_5fold_run.log" 2>&1
  echo "=== $(date) finished ${dataset} ${backbone} 5-fold ==="
}

run_busbra
run_kfold busi 224 efficientnet_b0 mmfm-busi
run_kfold breast 224 efficientnet_b0 mmfm-breast

# results-summary JSONs land in the repo root
mv -f ./busbra_5fold_resnet50_results.json ./busi_5fold_efficientnet_b0_results.json \
      ./breast_5fold_efficientnet_b0_results.json runs/run-2/extra-backbones/ 2>/dev/null || true

echo "=== $(date) ALL DONE ==="
