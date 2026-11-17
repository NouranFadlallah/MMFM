#!/bin/bash
# Fold-aligned fusion, all layers trained, with BUSC grouped by blocks of 10 consecutive
# frames (the duplicate-grouped runs still leak BUSC: test AUC 0.998). Otherwise identical
# to queue.sh step 2 (all).
cd /home/nouran/MMFM
LOG=runs/fold_aligned/busc_seq_fusion.log
for fold in 1 2 3 4 5; do
  echo "$(date '+%F %T') fold $fold" >> $LOG
  [ "$(df --output=avail /home | tail -1)" -gt 1500000 ] || { echo "STOP: disk" >> $LOG; exit 1; }
  .venv/bin/python training/train.py --dataset combined --combined-fold $fold --kfold 5 --pretrained \
    --epochs 10 --batch-size 32 --sequence-block-groups 10 \
    --mammography-init-checkpoint cbis_ddsm_single_resnet18_best.pth \
    --ultrasound-init-checkpoint 'runs/warm/busbra_fold{fold}.pth' \
    --mri-init-checkpoint runs/run-1/breamdm_single_resnet18_best.pth \
    --checkpoint-path runs/fold_aligned/fusion_seq_fold${fold}.pth \
    --wandb-project mmfm-combined --wandb-run-name fold-aligned-seq-f${fold} \
    > runs/fold_aligned/fusion_seq_fold${fold}.log 2>&1 || echo "fold $fold FAILED" >> $LOG
done
echo "$(date '+%F %T') ALL DONE" >> $LOG
