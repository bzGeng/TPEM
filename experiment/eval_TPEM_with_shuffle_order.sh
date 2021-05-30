#!/bin/bash
# Normally, bash shell cannot support floating point arthematic, thus, here we use `bc` package

readarray tasks < shuffle_task_order.txt
for task_id in `seq 1 7`; do
    tasks[$task_id]=`echo ${tasks[task_id]}`
done

GPU_ID=0

for task_id_1 in `seq 1 7`; do
   for task_id_2 in `seq 1 $task_id_1`; do
      CUDA_VISIBLE_DEVICES=$GPU_ID python ./eval_TPEM.py \
          -t ${tasks[task_id_2]} \
          -m test \
          -path save_prune/${tasks[task_id_1]}
    done
done
