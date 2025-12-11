export NCCL_SOCKET_IFNAME="lo"
export CUDA_VISIBLE_DEVICES=0,1,2,3

model=9.11-dinov3-ex=4-8-3-layer=24-finetune-4/ckpt_t22.002h.pt
# model=12.10-finetune-fl/ckpt_t0.100h.pt
# model=9.5-dinov3-ex=0-8-3-layer=24-scale/ckpt_242200.000000.pt
model=9.14-dinov3-ex=0-8-3-layer=24-scale-repastop/ckpt_0.000000.pt

torchrun --nproc_per_node 4 --nnodes 1 \
--rdzv_id 18642 --rdzv_backend c10d --rdzv_endpoint localhost:29512 \
inference.py --config configs/Efficient-LVSM.yaml \
    training.dataset_path = ../test/full_list.txt \
    training.batch_size_per_gpu = 12 \
    training.target_has_input =  false \
    training.num_input_views = 2 \
    training.num_target_views = 3 \
    inference.if_inference = true \
    inference.compute_metrics = true \
    inference.render_video = false \
    training.enable_repa = false \
    training.use_compile = false \
    inference.checkpoint_dir = experiments/checkpoints/$model \
    inference.inference_out_dir = experiments/evaluation/$model \
    model.transformer.n_layer = 12 \
    model.target_pose_tokenizer.image_size = 512 \
    model.image_tokenizer.image_size = 512 \
    