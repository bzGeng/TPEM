#!/bin/bash
# Normally, bash shell cannot support floating point arthematic, thus, here we use `bc` package

tasks=(
    'None'                # dummy
    'schedule'
    'navigate'
    'weather'
    'restaurant'
    'hotel'
    'attraction'
    'camrest'
)

GPU_ID=0

for task_id in `seq 1 7`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python ./eval_GLMP.py \
        -t ${tasks[task_id]} \
        -m test \
        -path save_train/${tasks[task_id]}
done
