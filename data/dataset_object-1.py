import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
import PIL.Image
import math
import random
import torch.nn.functional as F

class ObjaverseDataset(Dataset):
    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger
        self.objaverse_root_path = config.training.dataset_path
        self.inference = self.config.inference.get("if_inference", False)

        # 假设 objaverse_root_path 下是很多 UID 文件夹，如 '00a4a3597c584154a7c0d5c31f491b03/'
        # self.all_scene_uids = [uid for uid in os.listdir(self.objaverse_root_path) 
        #                        if os.path.isdir(os.path.join(self.objaverse_root_path, uid))]
        if hasattr(config.training, 'list_path'):
            with open(config.training.list_path, 'r') as f:
                self.uid_path_list = json.load(f)
        else:
            self.uid_path_list = {}
            for uid in os.listdir(self.objaverse_root_path):
                self.uid_path_list[uid] = os.path.join(self.objaverse_root_path, uid)
        # with open('data/valid_list.txt', 'r') as f:
            # valid = f.readlines()
        excludes = []
        if self.inference:
            excludes = os.listdir('dataset/gso_render_f')
        # print(valid)
        if self.inference:
            first_300_items = {k: v for i, (k, v) in enumerate(self.uid_path_list.items()) if i < 100}
            self.all_scene_uids = [uid for uid, path in first_300_items.items() if not uid in excludes]
        else:
            self.all_scene_uids = [uid for uid, path in self.uid_path_list.items() if not uid in excludes]
        self.all_scene_uids = self.all_scene_uids[:1]
        if self.inference:
            self.all_scene_uids = self.all_scene_uids
        print(len(self.all_scene_uids))

        # 渲染时使用的固定参数 (这是关键！)
        self.H, self.W = 512, 512
        # !!! 这里的 fovy 必须与你 Blender 渲染脚本中设置的 cam.data.angle_y 完全一致 !!!
        self.fovy_degrees = 49.1
        self.fovy_rad = math.radians(self.fovy_degrees)

        # 预先计算原始内参
        self.fx = self.fy = (self.H / 2) / math.tan(self.fovy_rad / 2)
        self.cx = self.W / 2
        self.cy = self.H / 2
        self.original_intrinsics = torch.tensor([self.fx, self.fy, self.cx, self.cy]).float()
        print('dataset len:', len(self.all_scene_uids))


    def __len__(self):
        return len(self.all_scene_uids)

    def preprocess_frames(self, view_indices, scene_dir):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        c2ws_blender = []
        intrinsics = []
        for view_idx in view_indices:
            image_path = os.path.join(scene_dir, f"{view_idx:03d}.png")
            image = PIL.Image.open(image_path).convert("RGB")
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            # if torch.distributed.get_rank() == 0:
            #     import ipdb; ipdb.set_trace()

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            fxfycxcy = self.original_intrinsics
            pose_path = os.path.join(scene_dir, f"{view_idx:03d}.npy")
            # Blender 输出是 3x4 的 RT 矩阵
            rt_matrix = np.load(pose_path)
            # 转换为 4x4 的 c2w 矩阵
            c2w = np.concatenate([rt_matrix, np.array([[0, 0, 0, 1]])], axis=0)
            c2ws_blender.append(torch.from_numpy(c2w).float())
            
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws_blender = torch.stack(c2ws_blender, dim=0)
        blender_to_opencv = torch.tensor([[1, 0, 0, 0],
                                          [0, -1, 0, 0],
                                          [0, 0, -1, 0],
                                          [0, 0, 0, 1]]).float()
        c2ws_opencv = c2ws_blender @ blender_to_opencv
        # w2cs = np.stack([np.array(frame["w2c"]) for frame in view_indices])
        # c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        # c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws_opencv
    
    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def __getitem__(self, idx):
        scene_uid = self.all_scene_uids[idx]
        scene_dir = self.uid_path_list[scene_uid]
        
        # 1. 选择视图
        num_total_views = len([f for f in os.listdir(scene_dir) if f.endswith('.png')])
        # print(num_total_views)
        # 随机选择 self.config.training.num_views 个视图
        # view_indices = random.sample(range(num_total_views), self.config.training.num_views)
        # view_indices.sort()
        # 定义4个分组，每组8个视图
        if self.inference:
            view_indices = [0,1,2,3]
            remaining = self.config.training.num_views - len(view_indices)
            if remaining > 0:
                # 从所有视图中选择剩余的（排除已选的）
                all_indices = list(range(num_total_views))
                available = [idx for idx in all_indices if idx not in view_indices]
                if len(available) >= remaining:
                    additional = random.sample(available, remaining)
                    view_indices.extend(additional)
                else:
                    view_indices.extend(available)
        elif num_total_views == self.config.training.num_views:
            view_indices = list(range(32))
            # print(1234)
        else:
            view_indices = [0,8,16,24,1,3,5,7,9,11,13,15]
        
        images = []
        intrinsics = []
        c2ws_blender = []

        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        for view_idx in view_indices:
            # --- 加载图像 ---
            image_path = os.path.join(scene_dir, f"{view_idx:03d}.png")
            image = PIL.Image.open(image_path).convert("RGB")
            
            # --- 图像和内参 Resize ---
            # 计算新的宽度，保持长宽比
            original_image_w, original_image_h = image.size
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            
            # 复制原始内参，避免修改类属性
            fxfycxcy = self.original_intrinsics.clone() 

            # 根据resize比例调整内参
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy[0] *= resize_ratio_x  # fx
            fxfycxcy[1] *= resize_ratio_y  # fy
            fxfycxcy[2] *= resize_ratio_x  # cx
            fxfycxcy[3] *= resize_ratio_y  # cy
            
            # 如果裁剪，进一步调整图像和主点
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_w = (resize_w - min_size) // 2
                start_h = (resize_h - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))
                fxfycxcy[2] -= start_w  # cx
                fxfycxcy[3] -= start_h  # cy

            image_tensor = torch.from_numpy(np.array(image) / 255.0).permute(2, 0, 1).float()
            
            images.append(image_tensor)
            intrinsics.append(fxfycxcy)

            # --- 加载外参 ---
            pose_path = os.path.join(scene_dir, f"{view_idx:03d}.npy")
            # try:
            rt_matrix = np.load(pose_path)
            # except Exception as e:
            # rt_matrix = np.ones((3,4), dtype=np.float32)
            c2w = np.concatenate([rt_matrix, np.array([[0, 0, 0, 1]])], axis=0)
            c2ws_blender.append(torch.from_numpy(c2w).float())

        # 3. 整合和后续处理
        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws_blender = torch.stack(c2ws_blender, dim=0)

        # --- 外参坐标系转换 (Blender -> OpenCV) ---
        # Blender: +X=右, +Y=上, +Z=后 (相机朝向-Z)
        # OpenCV:  +X=右, +Y=下, +Z=前 (相机朝向+Z)
        # 转换矩阵，用于翻转Y和Z轴
        blender_to_opencv = torch.tensor([[1, 0, 0, 0], 
                                        [0, -1, 0, 0], 
                                        [0, 0, -1, 0], 
                                        [0, 0, 0, 1]], 
                                        dtype=torch.float32, device=c2ws_blender.device)
        c2ws_opencv = c2ws_blender @ blender_to_opencv

        # 5. 应用姿态归一化
        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(c2ws_opencv, scene_scale_factor)

        # 6. 准备索引
        image_indices_tensor = torch.tensor(view_indices).long().unsqueeze(-1)
        scene_indices_tensor = torch.full_like(image_indices_tensor, idx)
        indices = torch.cat([image_indices_tensor, scene_indices_tensor], dim=-1)
        # print(input_c2ws[0])
        with open('data/valid_list.txt', 'a') as f:
            f.write(scene_uid+'\n')
        # print(f'index:{indices}')
        return {
            "image": images,
            "c2w": input_c2ws,
            "fxfycxcy": intrinsics,
            "index": indices,
            "scene_name": scene_uid
        }