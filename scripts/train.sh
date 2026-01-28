export NCCL_SOCKET_IFNAME="lo"
export CUDA_VISIBLE_DEVICES=0,1
# export WANDB_MODE=offline

exp_name=your_name
torchrun --nproc_per_node 2 --nnodes 1 \
    --rdzv_id 18636 --rdzv_backend c10d --rdzv_endpoint localhost:29503 \
    train.py --config configs/Efficient-LVSM.yaml \
    training.wandb_exp_name = $exp_name \
    training.checkpoint_dir = ./experiments/checkpoints/$exp_name \
    model.image_tokenizer.type = dinov3 \
    training.enable_repa = true \
    training.use_compile = true \
    training.batch_size_per_gpu = 2 \
    model.transformer.n_layer = 6 \
    model.transformer.attention_arch = flash \
    training.checkpoint_every_epoch = 5 \
    training.train_epochs = 100 \
    training.train_time = 24 \
    training.checkpoint_every_time = 1 \
    training.num_input_views = 2 \
    training.num_target_views = 6 \
    model.repa_config = 8-2 \
