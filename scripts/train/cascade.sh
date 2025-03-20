#!/bin/bash

export WANDB_ENTITY="[REDACTED]"

dataset_seed=42
export DATASET_SEED=$dataset_seed

num_strs=32
export NUM_STRS=$num_strs

epoch=2
export NUM_TRAIN_EPOCHS=$epoch

fix_rstr_loc="none"
export FIX_RSTR_LOC=$fix_rstr_loc

shuffle="1"
export DATALOADER_SHUFFLE=$shuffle

min_len=8
max_len=512
export RSTR_MIN_LEN=$min_len
export RSTR_MAX_LEN=$max_len

export CONTEXT_LEN=1024
export BATCH_SIZE=1024
export MICRO_BATCH_SIZE=32

for seed in 42 142857 2225393 20000308 2018011309; do
    export SEED=$seed

    for seq_overlap in 0 1; do
        export SEQ_OVERLAP=$seq_overlap
        for i in $(seq 13 13); do
            in_mode_each_occur=$(awk -v log_value="$i" 'BEGIN {print 2^log_value}')
            export IN_MODE_EACH_OCCUR=$in_mode_each_occur
                
            export RUN_NAME="CASCADE_ctx1024_dataseed${dataset_seed}_str${num_strs}_L${min_len}_${max_len}_occur${in_mode_each_occur}_epoch${epoch}_loc${fix_rstr_loc}_seed${seed}_shuffle${shuffle}_overlap${seq_overlap}_$(date +%m%d%H%M%S)"

            deepspeed --master_port=23333 --include="localhost:2,3,4,5" cascade_ds_train.py configs/phi1-162m_cascade.yaml
        done
    done
done