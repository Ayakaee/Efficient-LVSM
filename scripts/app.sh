export NCCL_SOCKET_IFNAME="lo"
export CUDA_VISIBLE_DEVICES=0
python app.py --config configs/Efficient-LVSM.yaml \
    training.dataset_path = data/demo.txt \
    training.batch_size_per_gpu = 12 \
    training.target_has_input =  false \
    training.square_crop = true \
    training.num_input_views = 6 \
    training.num_target_views = 1 \
    inference.if_inference = true \
    inference.compute_metrics = true \
    inference.render_video = false \
    training.enable_repa = false \
    training.use_compile = false \
    inference.checkpoint_dir = experiments/evaluation/9.11-dinov3-ex=4-8-3-layer=24-finetune-4/ckpt_t22.002h.pt \
    model.transformer.n_layer = 12 \
    inference.resize = false \
    model.target_pose_tokenizer.image_size = 512 \
    model.image_tokenizer.image_size = 512 \