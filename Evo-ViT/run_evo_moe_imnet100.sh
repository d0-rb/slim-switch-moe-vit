#!/bin/bash

MODEL=evo_moe_deit_tiny_patch16_224_expert8_sparse_init
CUDA=6
NUM_CUDA=1
DATASET=IMNET100
LR="5e-4"
EPOCHS=300
PORT=$((9000 + RANDOM % 1000))

CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.launch --nproc_per_node=${NUM_CUDA} --master_port=${PORT} --use_env \
                           main_deit.py --model ${MODEL} --data-set ${DATASET} \
                           --drop-path 0 --batch-size 256 --data-path ~/DATASET/ImageNet100 --epochs ${EPOCHS} --lr ${LR}\
                           --output_dir logs/imnet100/${MODEL}/lr_${LR}_ep_${EPOCHS}
