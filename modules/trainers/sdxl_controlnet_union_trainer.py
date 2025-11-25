import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from ..utils import sdxl_model_utils, sdxl_train_utils
from .sdxl_trainer import SDXLTrainer
from .sd15_controlnet_trainer import SD15ControlNetTrainer
from ..datasets.sdxl_image_condition_dataset import SDXLImageConditionDataset
from ..datasets.sdxl_union_json_dataset import SDXLUnionJSONDataset
from ..train_state.sdxl_controlnet_train_state import SDXLControlNetTrainState
from ..models.sdxl.controlnet_union import ControlNetModel_Union


class SDXLControlNetUnionTrainer(SDXLTrainer, SD15ControlNetTrainer):
    """Trainer for multi-condition ControlNet Union model.

    Reuses most logic from SDXLControlNetTrainer but swaps the ControlNetModel with ControlNetModel_Union
    and expects batch to provide:
        batch['condition_images_list']: list of tensors length = num_control_types (unused slots can be None)
        batch['union_control_type']: multi-hot tensor shape [B, num_control_types]
    For backward compatibility, if 'condition_images' exists and no list is given, we treat it as single control of index 0.
    """

    dataset_class = SDXLImageConditionDataset  # default; can switch to JSON driven
    nnet_class = UNet2DConditionModel
    pipeline_class = None  # not used in training loop sampling for now
    train_state_class = SDXLControlNetTrainState
    controlnet_class = ControlNetModel_Union

    def load_diffusion_model(self):
        return sdxl_model_utils.load_diffusers_models(
            self.pretrained_model_name_or_path,
            revision=self.revision,
            variant=self.variant,
            torch_dtype=self.weight_dtype,
            use_safetensors=self.use_safetensors,
            cache_dir=self.hf_cache_dir,
            token=self.hf_token,
            max_retries=self.max_retries,
            nnet_class=self.nnet_class,
        )

    def build_controlnet(self):
        # build ControlNetModel_Union from UNet config for compatibility
        # 使用 UNet config 中的 addition_time_embed_dim，若不存在则回退 256（SDXL 规范）
        addition_time_embed_dim = getattr(self.nnet.config, 'addition_time_embed_dim', 256) or 256
        controlnet = self.controlnet_class(
            in_channels=self.nnet.config.in_channels,
            conditioning_channels=3,
            flip_sin_to_cos=self.nnet.config.flip_sin_to_cos,
            freq_shift=self.nnet.config.freq_shift,
            down_block_types=tuple(self.nnet.config.down_block_types),
            only_cross_attention=self.nnet.config.only_cross_attention,
            block_out_channels=tuple(self.nnet.config.block_out_channels),
            layers_per_block=self.nnet.config.layers_per_block,
            downsample_padding=self.nnet.config.downsample_padding,
            mid_block_scale_factor=self.nnet.config.mid_block_scale_factor,
            act_fn=self.nnet.config.act_fn,
            norm_num_groups=self.nnet.config.norm_num_groups,
            norm_eps=self.nnet.config.norm_eps,
            cross_attention_dim=self.nnet.config.cross_attention_dim,
            transformer_layers_per_block=tuple(self.nnet.config.transformer_layers_per_block),
            encoder_hid_dim=self.nnet.config.encoder_hid_dim,
            encoder_hid_dim_type=self.nnet.config.encoder_hid_dim_type,
            attention_head_dim=tuple(self.nnet.config.attention_head_dim),
            num_attention_heads=self.nnet.config.num_attention_heads,
            use_linear_projection=self.nnet.config.use_linear_projection,
            class_embed_type=self.nnet.config.class_embed_type,
            addition_embed_type=self.nnet.config.addition_embed_type,
            num_class_embeds=self.nnet.config.num_class_embeds,
            upcast_attention=self.nnet.config.upcast_attention,
            resnet_time_scale_shift=self.nnet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=self.nnet.config.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order='rgb',
            conditioning_embedding_out_channels=(16,32,96,256),
            global_pool_conditions=False,
            addition_embed_type_num_heads=getattr(self.nnet.config, 'addition_embed_type_num_heads', 64),
            num_control_type=getattr(self.config, 'num_control_type', 6),
            addition_time_embed_dim=addition_time_embed_dim,
        )
        return controlnet.to(self.device, dtype=self.controlnet_dtype)

    def on_after_models_loaded(self):
        super().on_after_models_loaded()
        # Replace controlnet if not already built
        if not hasattr(self, 'controlnet') or self.controlnet is None or not isinstance(self.controlnet, ControlNetModel_Union):
            self.controlnet = self.build_controlnet()
        self.logger.info(f"Loaded ControlNet Union with {self.config.num_control_type if hasattr(self.config,'num_control_type') else 6} control types")
        # Switch dataset class if JSON manifest provided
        if getattr(self.config, 'union_json_path', None):
            self.logger.info("Using SDXLUnionJSONDataset via union_json_path")
            self.dataset_class = SDXLUnionJSONDataset

    def train_step(self, batch):
        # Prepare latents
        if batch.get('latents') is not None:
            latents = batch['latents'].to(self.device)
        else:
            with torch.no_grad():
                latents = self.vae.encode(batch['images'].to(self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
        latents *= self.vae_scale_factor

        target_size = batch['target_size_hw']
        orig_size = batch['original_size_hw']
        crop_top_lefts = batch['crop_top_lefts']
        prompt_embeds, unet_added_conditions = self.get_embeddings_diffusers(
            batch['captions'], target_size, orig_size, crop_top_lefts,
        )

        noise = self.get_noise(latents)
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.get_noisy_latents(latents, noise, timesteps).to(self.weight_dtype)

        # Retrieve multi condition list & control type vector
        union_control_type = batch.get('union_control_type')  # shape [B, num_control_type]
        if union_control_type is None:
            num_control_type = getattr(self.config, 'num_control_type', 6)
            control_type_1d = torch.zeros(num_control_type, device=self.device, dtype=self.weight_dtype)
            # 默认激活 canny 位置 index=3 与推理示例保持一致
            if num_control_type > 3:
                control_type_1d[3] = 1.0
            else:
                control_type_1d[0] = 1.0
            union_control_type = control_type_1d.unsqueeze(0)
        else:
            union_control_type = union_control_type.to(self.device)

        # 自动对齐长度到 in_features // addition_time_embed_dim
        at_dim = getattr(self.controlnet.config, 'addition_time_embed_dim', 256) or 256
        in_features = self.controlnet.control_add_embedding.linear_1.in_features
        need_types = in_features // at_dim
        have_types = union_control_type.shape[1]
        if have_types != need_types:
            if have_types < need_types:
                pad = need_types - have_types
                union_control_type = torch.cat([union_control_type, torch.zeros(union_control_type.shape[0], pad, device=union_control_type.device, dtype=union_control_type.dtype)], dim=1)
            else:
                union_control_type = union_control_type[:, :need_types]
        # repeat to batch size
        if union_control_type.shape[0] != batch['images'].shape[0]:
            union_control_type = union_control_type.repeat(batch['images'].shape[0], 1)

        condition_images_list = batch.get('condition_images_list')
        if condition_images_list is None and batch.get('condition_images') is not None:
            # expand single condition into list
            img = batch['condition_images'].to(self.device, dtype=self.controlnet.dtype)
            num_control_types = union_control_type.shape[1]  # [B, num_control_type]
            condition_images_list = [img] + [torch.zeros_like(img) for _ in range(num_control_types - 1)]

        # ensure tensor list all on device / dtype
        proc_list = []
        for i, ci in enumerate(condition_images_list):
            if ci is None or (isinstance(ci, (int,float)) and ci == 0):
                # create zero placeholder with expected shape (B,3,H,W) matching first non-none
                # find reference shape
                ref = next((x for x in condition_images_list if isinstance(x, torch.Tensor)), None)
                if ref is None:
                    raise ValueError('No valid condition image tensor in condition_images_list')
                proc_list.append(torch.zeros_like(ref))
            else:
                proc_list.append(ci.to(self.device, dtype=self.controlnet.dtype))

        # ControlNet Union forward
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={**unet_added_conditions, 'control_type': union_control_type},
            controlnet_cond_list=proc_list,
            return_dict=False,
        )

        with self.accelerator.autocast():
            model_pred = self.nnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
                down_block_additional_residuals=[s.to(dtype=self.weight_dtype) for s in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                return_dict=False,
            )[0]

        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = self.get_loss(model_pred, target, timesteps, batch)
        return loss
