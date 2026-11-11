#!/bin/bash
# BUSC pilot for the 1002-A proposal (H1): row-level vs duplicate-grouped 5-fold CV,
# same settings as runs/run-2 (ResNet-18, --paper-match, 100 epochs, batch 32, 128 px), 3 seeds.
cd /home/nouran/MMFM
OUT=runs/busc_dup
for seed in 42 43 44; do
  for mode in row dup; do
    extra=""; [ $mode = dup ] && extra="--duplicate-groups"
    .venv/bin/python training/train.py --dataset busc --single-mode --kfold 5 --paper-match \
      --epochs 100 --batch-size 32 --image-size 128 --seed $seed $extra \
      --checkpoint-path $OUT/busc_${mode}_s${seed}.pth --wandb-run-name busc-${mode}-s${seed} \
      > $OUT/busc_${mode}_s${seed}.log 2>&1 || { echo "FAILED $mode $seed"; exit 1; }
    mv busc_5fold_resnet18_results.json $OUT/busc_${mode}_s${seed}_results.json
    echo "done $mode seed $seed"
  done
done
