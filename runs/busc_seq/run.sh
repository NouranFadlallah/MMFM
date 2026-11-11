#!/bin/bash
# BUSC: 5-fold CV grouped by blocks of 10 consecutive file numbers (stand-in for patient
# IDs; frames are numbered in acquisition order). Same settings as runs/busc_dup, 3 seeds.
cd /home/nouran/MMFM
OUT=runs/busc_seq
for seed in 42 43 44; do
  .venv/bin/python training/train.py --dataset busc --single-mode --kfold 5 --paper-match \
    --epochs 100 --batch-size 32 --image-size 128 --pretrained --seed $seed --sequence-block-groups 10 \
    --wandb-project mmfm-busc --checkpoint-path $OUT/busc_seq10_s${seed}.pth --wandb-run-name busc-seq10-s${seed} \
    > $OUT/busc_seq10_s${seed}.log 2>&1 || { echo "FAILED seed $seed"; exit 1; }
  mv busc_5fold_resnet18_results.json $OUT/busc_seq10_s${seed}_results.json
  [ $seed != 42 ] && rm -f $OUT/busc_seq10_s${seed}_fold*.pth
  echo "$(date '+%F %T') done seed $seed"
done
