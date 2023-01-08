#!/bin/bash
read -p 'model: ' model #="resmoe_tiny_patch16_224_expert8"
read -p 'checkpoint file: ' resume
output=$(dirname $resume)
read -p 'cuda: ' cuda
read -p 'seed: ' n #=0
read -p 'num_samples: ' batch
read -p 'block depth (of gate to visualize): ' block_depth
read -p 'gate name (dense_gate, moe_gate): ' gate_name

CUDA_VISIBLE_DEVICES=$cuda

python3 skip_tok_vis.py --model $model --data-set CIFAR10 --data-path ./dataset --batch $batch --input-size 224 --eval-crop-ratio 1.0 --drop 0.0 --seed $n --drop-path 0.0 --gate-depth $block_depth --gate-name $gate_name --resume $resume --output_dir $output