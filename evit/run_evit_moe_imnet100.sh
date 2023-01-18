
# python3 -m torch.distributed.launch --nproc_per_node=3 --master_port=$RANDOM --use_env \

LR=5e-4
EPOCHS=300
MODEL=moe_deit_tiny_patch16_224_shrink_base_expert8
DATASET=IMNET100
CUDA=7

CUDA_VISIBLE_DEVICES=${CUDA} python main.py --model ${MODEL} \
  --fuse_token --data-set ${DATASET} \
	--base_keep_rate 0.7 \
	--input-size 224 \
	--batch-size 256 \
	--warmup-epochs 5 \
	--shrink_start_epoch 10 \
	--shrink_epochs 100 \
	--lr ${LR} \
	--epochs ${EPOCHS} \
	--dist-eval \
	--data-path ~/DATASET/ImageNet100 \
	--output_dir logs/imnet100/${MODEL}/lr_${LR}_ep_${EPOCHS}
