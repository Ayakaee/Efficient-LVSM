# Copyright (c) 2025 Haian Jin. Original LVSM implementation (ICLR 2025).
# Copyright (c) 2025 Yihang Sun. Modifications for Efficient-LVSM.
#
# This code is based on the LVSM project by Haian Jin et al.
# Original repository: https://github.com/Haian-Jin/LVSM
# 
# Licensed under CC BY-NC-SA 4.0 - see LICENSE.md for details.

import os
import sys
import traceback

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from easydict import EasyDict as edict
from torchvision.transforms import Normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import camera_utils, data_utils 
from model.transformer import QK_Norm_SelfAttentionBlock, QK_Norm_SelfCrossAttentionBlock, init_weights
from model.loss import LossComputer
from model.repa_config import repa_map
from utils.training_utils import format_number


# Constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


class Images2LatentScene(nn.Module):
    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)
        self.logger = logger

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()
        
        # Initialize transformer blocks
        self._init_transformer()
        
        # Initialize loss computer
        self.loss_computer = LossComputer(config)

        # Initialize REPA
        if self.config.training.enable_repa:
            self._init_repa()
            
        # Initialize KV cache for incremental inference
        self.kv_cache = {}  # Stores Key and Value for each layer        
            
    def _init_repa(self):
        """Initialize REPA (REPresentation Alignment) components."""
        # Determine feature dimension based on encoder type
        encoder_type = self.config.model.image_tokenizer.type
        if 'dino' in encoder_type:
            z_dim = 768
        else:
            raise NotImplementedError(f"Unknown image tokenizer type: {encoder_type}")
        
        # Initialize REPA structures
        self.repa_label = {'input': {}, 'target': {}}
        self.repa_x = {'input': {}, 'target': {}}
        self.repa_config = repa_map[self.config.model.repa_config]
        
        # Create projector modules
        self.repa_projector = nn.ModuleDict({
            'input': nn.ModuleDict(),
            'target': nn.ModuleDict()
        })
        
        for repa_type in ['input', 'target']:
            for key, value in self.repa_config[repa_type].items():
                projector_dict = nn.ModuleDict()
                self.repa_label[repa_type][key] = None
                for idx in value:
                    self.repa_x[repa_type][idx] = None
                    projector_dict[str(idx)] = self._create_repa_projector(
                        self.config.model.transformer.d, 
                        self.config.model.projector_dim, 
                        z_dim
                    )
                self.repa_projector[repa_type][str(key)] = projector_dict

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """
        Create a tokenizer that converts images to tokens.
        
        Args:
            in_channels: Number of input channels
            patch_size: Size of each patch
            d_model: Model dimension
            
        Returns:
            nn.Sequential: Tokenizer module
        """
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)
        return tokenizer
    
    def _create_repa_projector(self, d_model, hidden_dim, output_dim):
        """
        Create a REPA projector with 3-layer MLP.
        
        Args:
            d_model: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            
        Returns:
            nn.Sequential: Projector module
        """
        return nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        config = self.config.model.image_tokenizer
        self.rgbp_tokenizer = self._create_tokenizer(
            in_channels = config.in_channels,
            patch_size = config.patch_size,
            d_model = self.config.model.transformer.d
        )
        if config.type == 'dinov2':
            import timm
            if config.source == 'local':
                encoder = torch.hub.load(config.model_source_dir, 'dinov2_vitb14', source='local')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            del encoder.head
            patch_resolution = config.image_size // config.patch_size
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            self.repa_encoder = encoder
        elif config.type == 'dinov3':
            import timm
            encoder = torch.hub.load(
                repo_or_dir='dinov3',
                model='dinov3_vitb16',
                source='local',
                pretrained=False
            )
            state_dict = torch.load(config.model_path, map_location="cpu")
            encoder.load_state_dict(state_dict)
            # patch_resolution = config.image_size // config.patch_size
            # encoder.rope_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            #     encoder.rope_embed.data, [patch_resolution, patch_resolution],
            # )
            encoder.head = torch.nn.Identity()
            self.repa_encoder = encoder
        elif config.type == 'none':
            self.repa_encoder = None
        else:
            raise NotImplementedError('unknown enocder type')
            
        self.align_projector = None
            
        if self.repa_encoder is not None:
            freeze_encoder = self.config.model.get("freeze_repa_encoder", True)
            if freeze_encoder:
                for param in self.repa_encoder.parameters():
                    param.requires_grad = False
                self.repa_encoder.eval()
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.target_pose_tokenizer.in_channels,
            patch_size = self.config.model.target_pose_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, elementwise_affine=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(init_weights)

    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)
        use_flex_attention = config.attention_arch == 'flex'

        self.self_cross_blocks = nn.ModuleList([
            QK_Norm_SelfCrossAttentionBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
            ) for _ in range(config.n_layer)
        ])

        self.input_self_attn_blocks = nn.ModuleList([
            QK_Norm_SelfAttentionBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
            ) for _ in range(config.n_layer)
        ])
        # Apply special initialization if configured
        if config.get("special_init", False):
            # Initialize self-cross blocks
            for idx, block in enumerate(self.self_cross_blocks):
                if config.depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            for block in self.self_cross_blocks:
                block.apply(init_weights)
            
        # self.transformer_input_layernorm = nn.LayerNorm(config.d, elementwise_affine=False)

    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()

    def forward_features(self, x, output_layer=None, repa_type=None, masks=None):
        x = self.repa_encoder.prepare_tokens_with_masks(x, masks)

        for idx, blk in enumerate(self.repa_encoder.blocks):
            x = blk(x)
            if repa_type is not None:
                if idx + 1 in self.repa_label[repa_type].keys():
                    self.repa_label[repa_type][idx + 1] = x[:, 1:, :]

        x_norm = self.repa_encoder.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.repa_encoder.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.repa_encoder.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward_features_v3(self, x_list, masks_list, repa_type=None):
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.repa_encoder.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for idx, blk in enumerate(self.repa_encoder.blocks):
            if self.repa_encoder.rope_embed is not None:
                rope_sincos = [self.repa_encoder.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
            if repa_type is not None:
                if idx + 1 in self.repa_label[repa_type].keys():
                    self.repa_label[repa_type][idx + 1] = x[0][:, 5:, :]
            # x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.repa_encoder.untie_cls_and_patch_norms or self.repa_encoder.untie_global_and_local_cls_norm:
                if self.repa_encoder.untie_global_and_local_cls_norm and self.repa_encoder.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.repa_encoder.local_cls_norm(x[:, : self.repa_encoder.n_storage_tokens + 1])
                elif self.repa_encoder.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.repa_encoder.cls_norm(x[:, : self.repa_encoder.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.repa_encoder.norm(x[:, : self.repa_encoder.n_storage_tokens + 1])
                x_norm_patch = self.repa_encoder.norm(x[:, self.repa_encoder.n_storage_tokens + 1 :])
            else:
                x_norm = self.repa_encoder.norm(x)
                x_norm_cls_reg = x_norm[:, : self.repa_encoder.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.repa_encoder.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def get_repa_feature(self, image, repa_type):
        """
        Extract REPA features from images using pretrained encoder.
        
        Args:
            image: Input images [B, V, C, H, W]
            repa_type: Type of REPA feature ('input' or 'target')
        """
        with torch.no_grad():
            enc_type = self.config.model.image_tokenizer.type
            x = rearrange(image, "b v c h w -> (b v) c h w")
            
            # Normalize and extract features based on encoder type
            x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
            
            if 'dinov2' in enc_type:
                x = torch.nn.functional.interpolate(x, size=336, mode='bicubic')
                x = self.forward_features(x, None, repa_type=repa_type)
                x = x['x_norm_patchtokens']
            elif 'dinov3' in enc_type:
                x = torch.nn.functional.interpolate(x, size=512, mode='bicubic')
                x = self.forward_features_v3([x], [None], repa_type=repa_type)[0]
                x = x['x_norm_patchtokens']
            else:
                raise NotImplementedError(f"Invalid image tokenizer type: {enc_type}")
            
            # Resize features to target grid size
            current_grid_size = int(x.shape[1] ** 0.5)
            target_grid_size = 32
            
            for idx in self.repa_label[repa_type].keys():
                x = self.repa_label[repa_type][idx]
                x = rearrange(x, "b (h w) d -> b d h w", h=current_grid_size, w=current_grid_size)
                x = torch.nn.functional.interpolate(
                    x, 
                    size=(target_grid_size, target_grid_size), 
                    mode='bicubic', 
                    align_corners=False
                )
                self.repa_label[repa_type][idx] = rearrange(x, "b d h w -> b (h w) d")

    def generate_target_views(self, input_tokens, target_tokens, token_shape, use_kv_cache=False):
        """
        Independent target generator that uses cached input tokens for cross-attention.
        
        Args:
            target_tokens: Target pose tokens [B, v_target, D]
            token_shape: (v_input, v_target, n_patches) Token shape information
            use_kv_cache: Whether to use KV cache
            
        Returns:
            generated_target_tokens: Generated target tokens
        """
        v_input, v_target, n_patches = token_shape
        bv, _, d = target_tokens.shape
        b = bv // v_target
        
        for idx, block in enumerate(self.self_cross_blocks):
            input_tokens = input_tokens.view(b * v_input, n_patches, d)
            input_tokens = self.input_self_attn_blocks[idx](input_tokens)
            input_tokens = input_tokens.view(b, v_input * (n_patches), d)
            if use_kv_cache and f'self_cross_{idx}' in self.kv_cache:
                # Use cached KV for cross-attention computation
                cached_kv = self.kv_cache[f'self_cross_{idx}']
                target_tokens, kv_cache = block(input_tokens, target_tokens, past_kv=cached_kv, return_kv=True)
                self.kv_cache[f'self_cross_{idx}'] = kv_cache
            else:
                target_tokens, kv_cache = block(input_tokens, target_tokens, return_kv=True)
                if use_kv_cache:
                    self.kv_cache[f'self_cross_{idx}'] = kv_cache
        
        return target_tokens

    def pass_layers(self, input_tokens, target_tokens, token_shape):
        """
        Pass through transformer layers.
        
        Args:
            input_tokens: [B, v_input, n_patches, D]
            target_tokens: [B, v_target, n_patches, D]
            token_shape: (v_input, v_target, n_patches) Token shape information
        """
        # Get token shape information
        v_input, v_target, n_patches = token_shape
        bv, _, d = input_tokens.shape
        b = bv // v_input
        
        # Self-Cross
        for idx, block in enumerate(self.self_cross_blocks):
            input_tokens = input_tokens.view(b * v_input, n_patches, d)
            input_tokens = self.input_self_attn_blocks[idx](input_tokens)
            if self.config.training.enable_repa:
                if idx + 1 in self.repa_x['input']:
                    self.repa_x['input'][idx + 1] = input_tokens
            input_tokens = input_tokens.view(b, v_input * (n_patches), d)
            target_tokens = block(input_tokens, target_tokens)
                
            if self.config.training.enable_repa:
                if idx + 1 in self.repa_x['target']:
                    self.repa_x['target'][idx + 1] = target_tokens
    
        return target_tokens
    
    def clear_kv_cache(self):
        """Clear KV cache for incremental inference."""
        self.kv_cache = {}
        
    def incremental_forward(self, new_input_data, target_data, has_target_image=True, train=True, extract_features=False):
        """
        Incremental forward inference.
        
        Args:
            new_input_data: New input data (single or multiple views)
            target_data: Target data
            has_target_image: Whether target image exists
            train: Whether in training mode
            
        Returns:
            result: Inference result
        """
        posed_input_images = self.get_posed_input(
            images=new_input_data.image, ray_o=new_input_data.ray_o, ray_d=new_input_data.ray_d
        )
        b, v_input, c, h, w = posed_input_images.size()
        v_target = target_data.image.shape[1]
        rgbp_token = self.rgbp_tokenizer(posed_input_images)  # [b*v, n_patches, d]
        bv, n_patches, d = rgbp_token.size()
        
        if self.repa_encoder is not None and self.config.training.enable_repa:
            input_img_tokens = rgbp_token
            self.get_repa_feature(new_input_data.image, 'input')
            self.get_repa_feature(target_data.image, 'target')
        else:
            input_img_tokens = rgbp_token
        # update KV cache
        token_shape = (v_input, v_target, n_patches)  # assume target view is 1
        
        # process target pose
        target_pose_cond = self.get_posed_input(ray_o=target_data.ray_o, ray_d=target_data.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)
        
        # Use cached input tokens to generate target tokens
        target_image_tokens = self.generate_target_views(
            input_img_tokens, target_pose_tokens, token_shape, use_kv_cache=True
        )
        
        # Decode target image
        rendered_images = self.image_token_decoder(target_image_tokens)
        height, width = target_data.image_h_w
        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=3
        )

        if has_target_image:
            if self.config.training.enable_repa:
                loss_metrics = self.loss_computer(
                    rendered_images,
                    target_data.image,
                    self.repa_x,
                    self.repa_label,
                    self.repa_projector,
                    self.repa_config,
                    self.config.training.enable_repa,
                    train=(train and self.config.training.enable_repa)
                )
            else:
                loss_metrics = self.loss_computer(
                    rendered_images,
                    target_data.image,
                    train=train
                )
        else:
            loss_metrics = None

        result = edict(
            input=new_input_data,
            target=target_data,
            loss_metrics=loss_metrics,
            render=rendered_images,
        )
        
        return result
            
    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        """
        Create pose-conditioned input using Plucker coordinates.
        
        Args:
            images: Input images [B, V, C, H, W] or None
            ray_o: Ray origins [B, V, 3, H, W]
            ray_d: Ray directions [B, V, 3, H, W]
            method: Plucker embedding method ('default_plucker', 'custom_plucker', 'aug_plucker')
            
        Returns:
            Posed input [B, V, C+6, H, W] if images provided, else [B, V, 6, H, W]
        """
        if method == "custom_plucker":
            # Custom Plucker: [ray_d, nearest_point]
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            # Augmented Plucker: [o_cross_d, ray_d, nearest_point]
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            # Default Plucker: [o_cross_d, ray_d]
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            # Concatenate normalized images with pose conditioning
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)
    
    
    def forward(self, input, target, has_target_image=True, train=True, incremental_mode=False):
        """
        Forward inference method.
        
        Args:
            input: Input data
            target: Target data
            has_target_image: Whether target image exists
            train: Whether in training mode
            incremental_mode: Whether to use incremental inference mode
        """
        if incremental_mode:
            self.config.training.num_input_views = input.image.shape[1]
            self.config.training.num_target_views = target.image.shape[1]
            return self.incremental_forward(input, target, has_target_image, train)

        # Tokenize input images with pose conditioning
        posed_input_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        b, v_input, c, h, w = posed_input_images.size()
        rgbp_token = self.rgbp_tokenizer(posed_input_images)  # [B*V, N_patches, D]
        bv, n_patches, d = rgbp_token.size()

        # Extract REPA features if enabled
        if self.repa_encoder is not None and self.config.training.enable_repa:
            input_img_tokens = rgbp_token
            self.get_repa_feature(input.image, 'input')
            self.get_repa_feature(target.image, 'target')
        else:
            input_img_tokens = rgbp_token
        
        # Tokenize target pose
        target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)  # [B*V, N_patches, D]

        # Pass through transformer layers
        token_shape = (v_input, v_target, n_patches)
        target_image_tokens = self.pass_layers(input_img_tokens, target_pose_tokens, token_shape)

        # [b * v_target, n_patches, d] -> [b * v_target, n_patches, p*p*3]
        rendered_images = self.image_token_decoder(target_image_tokens)
        
        # Reshape decoded tokens to image format
        height, width = target.image_h_w
        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=3
        )
        
        # Compute losses if target images available
        if has_target_image:
            if self.config.training.enable_repa:
                loss_metrics = self.loss_computer(
                    rendered_images,
                    target.image,
                    self.repa_x,
                    self.repa_label,
                    self.repa_projector,
                    self.repa_config,
                    self.config.training.enable_repa,
                    train=(train and self.config.training.enable_repa)
                )
            else:
                loss_metrics = self.loss_computer(
                    rendered_images,
                    target.image,
                    train=train
                )
        else:
            loss_metrics = None

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images,
            )
        
        return result

    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """
        Render a video from the model.
        
        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses
            
        Returns:
            result: Updated with video rendering
        """
    
        if data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target

        # Prepare input tokens; [b, v, 3+6, h, w]
        posed_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        bs, v_input, c, h, w = posed_images.size()

        input_img_tokens = self.rgbp_tokenizer(posed_images)  # [b*v_input, n_patches, d]

        _, n_patches, d = input_img_tokens.size()  # [b*v_input, n_patches, d]
        # input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)  # [b, v_input*n_patches, d]
        # target_pose_cond_list = []
        if traj_type == "interpolate":
            c2ws = input.c2w # [b, v, 4, 4]
            fxfycxcy = input.fxfycxcy #  [b, v, 4]
            device = input.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device) # [b, v, 3, 3]
            intrinsics[:, :,  0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :,  1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :,  0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :,  1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4], intrinsics[b], num_frames, order_poses=order_poses
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0) # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0) # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        # Compute rays for rendering
        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(
            fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
        )

        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device), 
            ray_d=rendering_ray_d.to(input.image.device)
        )
                
        _, num_views, c, h, w = target_pose_cond.size()
    
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [bs*v_target, n_patches, d]
        _, n_patches, d = target_pose_tokens.size()  # [b*v_target, n_patches, d]
        all_target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)  # [b, v_target*n_patches, d]
        b, _, c, h, w = target_pose_cond.size()
        
        view_chunk_size = 4

        video_rendering_list = []
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)
            v_target = cur_view_chunk_size
            token_shape = (v_input, v_target, n_patches)
            # [b, (v_input*n_patches), d] -> [(b * cur_v_target), (v_input*n_patches), d]
            # repeated_input_img_tokens = repeat(input_img_tokens.detach(), 'b np d -> (b chunk) np d', chunk=cur_view_chunk_size, np=n_patches* v_input)

            start_idx, end_idx = cur_chunk * n_patches, (cur_chunk + cur_view_chunk_size) * n_patches            
            # [b, v_target * n_patches, d] -> [b, cur_v_target*n_patches, d] -> [b*cur_v_target, n_patches, d]
            # cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ], 
                                            #    "b (v_chunk p) d -> (b v_chunk) p d", 
                                            #    v_chunk=cur_view_chunk_size, p=n_patches)
            target_pose_tokens = all_target_pose_tokens[:, start_idx:end_idx,: ]
            target_pose_tokens = target_pose_tokens.reshape(b * v_target, n_patches, d)
            # cur_concat_input_tokens = torch.cat((repeated_input_img_tokens, cur_target_pose_tokens,), dim=1) # [b*cur_v_target, v_input*n_patches+n_patches, d]
            # cur_concat_input_tokens = self.transformer_input_layernorm(
                # cur_concat_input_tokens
            # )
            target_image_tokens = self.pass_layers(
                input_img_tokens, target_pose_tokens, token_shape)

            # transformer_output_tokens = self.pass_layers(cur_concat_input_tokens, gradient_checkpoint=False)

            # _, pred_target_image_tokens = transformer_output_tokens.split(
                # [v_input * n_patches, n_patches], dim=1
            # ) # [b * v_target, v*n_patches, d], [b * v_target, n_patches, d]
            
            height, width = target.image_h_w

            patch_size = self.config.model.target_pose_tokenizer.patch_size

            # [b, v_target*n_patches, p*p*3]
            video_rendering = self.image_token_decoder(target_image_tokens)
            
            video_rendering = rearrange(
                video_rendering, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=height // patch_size, 
                w=width // patch_size, 
                p1=patch_size, 
                p2=patch_size, 
                c=3
            ).cpu()

            video_rendering_list.append(video_rendering)
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering


        return data_batch

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if (file_name.endswith(".pt") and not 'ckpt_t' in file_name)]
            ckpt_names = sorted(ckpt_names, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
            print(f'load checkpoint from {ckpt_paths[-1]}')
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        state_dict = checkpoint["model"]
        if not self.config.training.use_compile:
            print("discard _orig_mod. in loading model")
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        return 0


