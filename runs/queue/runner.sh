#!/bin/bash
# GPU job queue: runs jobs.txt top line first, one at a time; edit jobs.txt freely.
# A line "STOP" ends the runner. Each job logs to logs/<n>.log; done.txt records results.
cd /home/nouran/MMFM
Q=runs/queue; mkdir -p $Q/logs
while pgrep -f "reproduce_odelia.py train --fold 4" >/dev/null; do sleep 60; done
n=0
while true; do
  job=$(head -n 1 $Q/jobs.txt 2>/dev/null)
  if [ -z "$job" ]; then sleep 60; continue; fi
  [ "$job" = "STOP" ] && exit 0
  sed -i '1d' $Q/jobs.txt
  n=$((n+1)); log=$Q/logs/$(date +%m%d_%H%M)_$n.log
  echo "$(date '+%F %T') START $job" >> $Q/done.txt
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash -c "$job" > $log 2>&1
  echo "$(date '+%F %T') EXIT $? $job -> $log" >> $Q/done.txt
done
