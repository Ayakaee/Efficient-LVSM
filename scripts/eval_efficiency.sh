export NCCL_SOCKET_IFNAME="lo"
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id 18641 --rdzv_backend c10d --rdzv_endpoint localhost:29508 \
analyse_inc.py --config configs/Efficient-LVSM.yaml \
    training.dataset_path = data/test/full_list.txt \
    model.class_name = model.efficient_lvsm.Images2LatentScene \
    inference.use_incremental_inference = true \
    training.batch_size_per_gpu = 1 \
    training.target_has_input =  false \
    training.num_input_views = 32 \
    training.num_target_views = 4 \
    inference.if_inference = true \
    inference.compute_metrics = false \
    inference.render_video = false \
    training.enable_repa = false \
    training.use_compile = false \
    inference.checkpoint_dir = experiments/checkpoints/9.14-dinov3-ex=0-8-3-layer=24-scale-repastop/ckpt_0.000000.pt \
    model.transformer.n_layer = 12 \
    inference.resize = false \
