export NCCL_SOCKET_IFNAME="lo"
export CUDA_VISIBLE_DEVICES=0

model=efficient_lvsm_res256

torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id 18642 --rdzv_backend c10d --rdzv_endpoint localhost:29256 \
inference_incremental.py --config configs/Efficient-LVSM.yaml \
    training.dataset_path = ../backup/test/full_list.txt \
    training.batch_size_per_gpu = 12 \
    training.target_has_input =  false \
    training.num_input_views = 2 \
    training.num_target_views = 3 \
    inference.if_inference = true \
    inference.compute_metrics = true \
    inference.render_video = false \
    training.enable_repa = false \
    training.use_compile = false \
    inference.checkpoint_dir = efficient_lvsm_res256.pt \
    inference.inference_out_dir = experiments/evaluation/$model \
    model.transformer.n_layer = 12 \
    model.target_pose_tokenizer.image_size = 256 \
    model.image_tokenizer.image_size = 256 \
    