#!/bin/bash
# Reviewer weak points (docs/paper_tik2_todo.md, A1): one GPU queue, run in order.
#  1. BUSC grouped by blocks of 10 consecutive frames (3 seeds)
#  2. fold-aligned fusion, 5 folds x {all trained, frozen backbones}, leak-free warm-starts:
#     mammography CBIS-DDSM (Shen split, test held out), ultrasound BUS-BRA fold-k model
#     (official K5B, trained without fold k's test), MRI BreastDM (official split)
#  3. BUSI row-level vs duplicate-grouped 5-fold CV (3 seeds)
#  4. single-dataset specialists on the same splits, for scripts/compare_fusion_specialists.py
# Every step keeps its results JSON / prediction NPZ; specialist checkpoints are deleted.
cd /home/nouran/MMFM
PY=.venv/bin/python
LOG=runs/fold_aligned/queue.log
step() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
disk_ok() {
  local free_kb=$(df --output=avail /home | tail -1)
  [ "$free_kb" -gt 1500000 ] || { step "STOP: less than 1.5 GB free"; exit 1; }
}

step "1. BUSC sequence-block CV"; disk_ok
bash runs/busc_seq/run.sh >> $LOG 2>&1 || step "busc_seq FAILED"

for variant in all frozen; do
  extra=""; [ $variant = frozen ] && extra="--freeze-backbones"
  for fold in 1 2 3 4 5; do
    step "2. fusion $variant fold $fold"; disk_ok
    $PY training/train.py --dataset combined --combined-fold $fold --kfold 5 --pretrained \
      --epochs 10 --batch-size 32 $extra \
      --mammography-init-checkpoint cbis_ddsm_single_resnet18_best.pth \
      --ultrasound-init-checkpoint 'runs/warm/busbra_fold{fold}.pth' \
      --mri-init-checkpoint runs/run-1/breamdm_single_resnet18_best.pth \
      --checkpoint-path runs/fold_aligned/fusion_${variant}_fold${fold}.pth \
      --wandb-project mmfm-combined --wandb-run-name fold-aligned-${variant}-f${fold} \
      > runs/fold_aligned/fusion_${variant}_fold${fold}.log 2>&1 || step "fusion $variant fold $fold FAILED"
  done
done

step "3. BUSI row vs dup"; disk_ok
bash runs/busi_dup/run.sh >> $LOG 2>&1 || step "busi_dup FAILED"

OUT=runs/specialists
kfold_specialist() {  # dataset, extra args
  local ds=$1; shift
  step "4. specialist $ds"; disk_ok
  $PY training/train.py --dataset $ds --single-mode --kfold 5 --paper-match --pretrained \
    --epochs 100 --batch-size 32 --seed 42 "$@" \
    --checkpoint-path $OUT/${ds}.pth --wandb-project mmfm-specialists --wandb-run-name spec-${ds} \
    > $OUT/${ds}.log 2>&1 || step "specialist $ds FAILED"
  mv ${ds}_5fold_*_results.json $OUT/${ds}_results.json 2>/dev/null
  rm -f $OUT/${ds}_fold*.pth
}
kfold_specialist mias
kfold_specialist breast
kfold_specialist busc --image-size 128 --duplicate-groups
kfold_specialist dmid
kfold_specialist bus_uclm
kfold_specialist bcsdbt_full
kfold_specialist bus_cot

for ds in breamdm odelia ambl cbis_ddsm; do  # fixed official splits, one model each
  step "4. specialist $ds"; disk_ok
  $PY training/train.py --dataset $ds --single-mode --pretrained --epochs 30 --batch-size 32 --seed 42 \
    --checkpoint-path $OUT/${ds}.pth --wandb-project mmfm-specialists --wandb-run-name spec-${ds} \
    > $OUT/${ds}.log 2>&1 || step "specialist $ds FAILED"
  rm -f $OUT/${ds}.pth
done
step "ALL DONE"
