#!/bin/bash
# BUSI: row-level vs duplicate-grouped 5-fold CV (reviewer weak point: leaky splits).
# Same settings as runs/run-2 (ResNet-18, --paper-match, 100 epochs, batch 32, 224 px), 3 seeds.
cd /home/nouran/MMFM
OUT=runs/busi_dup
for seed in 42 43 44; do
  for mode in row dup; do
    extra=""; [ $mode = dup ] && extra="--duplicate-groups"
    .venv/bin/python training/train.py --dataset busi --single-mode --kfold 5 --paper-match \
      --epochs 100 --batch-size 32 --image-size 224 --backbone resnet18 --pretrained --seed $seed $extra \
      --wandb-project mmfm-busi --checkpoint-path $OUT/busi_${mode}_s${seed}.pth --wandb-run-name busi-${mode}-s${seed} \
      > $OUT/busi_${mode}_s${seed}.log 2>&1 || { echo "FAILED $mode $seed"; exit 1; }
    mv busi_5fold_resnet18_results.json $OUT/busi_${mode}_s${seed}_results.json
    [ $seed != 42 ] && rm -f $OUT/busi_${mode}_s${seed}_fold*.pth
    echo "$(date '+%F %T') done $mode seed $seed"
  done
done
