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
lr=0.001
layer=3
hidden_size=128
dropout=0.2
batch_size=32
seed=3
dec=GLMP

for task_id in `seq 1 7`; do
      CUDA_VISIBLE_DEVICES=$GPU_ID python ./run_GLMP.py \
          -lr $lr \
          -l $layer \
          -m train \
          -t ${tasks[task_id]} \
          -hdd $hidden_size \
          -dr $dropout \
          -dec $dec \
          -bsz $batch_size \
          -seed $seed
done
