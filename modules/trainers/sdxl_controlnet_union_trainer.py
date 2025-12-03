import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from ..utils import sdxl_model_utils, sdxl_train_utils
from .sdxl_trainer import SDXLTrainer
from .sd15_controlnet_trainer import SD15ControlNetTrainer
from ..datasets.sdxl_image_condition_dataset import SDXLImageConditionDataset
from ..datasets.sdxl_union_json_dataset import SDXLUnionJSONDataset
from ..train_state.sdxl_controlnet_train_state import SDXLControlNetTrainState
from ..models.sdxl.controlnet_union import ControlNetModel_Union
from ..pipelines.sdxl_controlnet_union_pipeline import StableDiffusionXLControlNetUnionPipeline


# Wrapper pipeline to keep the trainer as the single compatibility surface.
# The original pipeline expects `image_list` name; external callers (eval_utils)
# send `image` or `controlnet_image`. Expose those names here and map them to
# `image_list` so inspect.signature sees `image` and eval_utils will pass it.
class _UnionPipelineWrapper(StableDiffusionXLControlNetUnionPipeline):
    def __call__(
        self,
        prompt=None,
        prompt_2=None,
        image=None,
        image_list=None,
        controlnet_image=None,
        height=None,
        width=None,
    union_control_type=None,
        **kwargs,
    ):
        if image_list is None:
            if image is not None:
                image_list = [image]
            elif controlnet_image is not None:
                image_list = controlnet_image
            else:
                image_list = [] 

        if union_control_type is None:
            try:
                # 尝试从 config 读取 num_control_type
                num_control_type = int(getattr(self.config, 'num_control_type', 6))
            except Exception:
                num_control_type = 6
            
            import torch as _torch
            union_control_type = _torch.zeros((1, num_control_type), dtype=_torch.float32)

            

        # 3. 调用父类
        return super().__call__(
            prompt=prompt,
            prompt_2=prompt_2,
            image_list=image_list,
            height=height,
            width=width,
            union_control_type=union_control_type,
            **kwargs,
        )


class SDXLControlNetUnionTrainer(SD15ControlNetTrainer,SDXLTrainer):
    def load_controlnet_model(self):
        """Override parent logic: always build ControlNet with correct config."""
        self.logger.info("  LOGGER-[UnionTrainer] Building ControlNet with build_controlnet (no from_unet)")
        controlnet = self.build_controlnet()
        return {"controlnet": controlnet}
    """Trainer for multi-condition ControlNet Union model.

    Reuses most logic from SDXLControlNetTrainer but swaps the ControlNetModel with ControlNetModel_Union
    and expects batch to provide:
        batch['condition_images_list']: list of tensors length = num_control_types (unused slots can be None)
        batch['union_control_type']: multi-hot tensor shape [B, num_control_types]
    For backward compatibility, if 'condition_images' exists and no list is given, we treat it as single control of index 0.
    """

    # Use the JSON-driven union dataset by default to avoid single-condition samplers
    dataset_class = SDXLUnionJSONDataset
    nnet_class = UNet2DConditionModel
    # Point to the new, correct pipeline wrapper for evaluation
    pipeline_class = _UnionPipelineWrapper
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

    # def build_controlnet(self):
    #     # Build a ControlNet that mirrors the UNet's architecture but with corrected parameters.
    #     self.logger.info("  LOGGER--- Calling build_controlnet ---")

    #     nnet_config = self.nnet.config

    #     # 1. Extract architecture directly from the provided UNet config.
    #     addition_embed_type = nnet_config.addition_embed_type
    #     down_block_types = nnet_config.down_block_types
    #     block_out_channels = nnet_config.block_out_channels
    #     layers_per_block = nnet_config.layers_per_block
    #     cross_attention_dim = nnet_config.cross_attention_dim
    #     transformer_layers_per_block = nnet_config.transformer_layers_per_block
    #     projection_class_embeddings_input_dim = nnet_config.projection_class_embeddings_input_dim

    #     # This is the key fix: the ControlNet must have this set to False.
    #     use_linear_projection = False

    #     self.logger.info(f"  LOGGER- UNet addition_embed_type: {addition_embed_type}")
    #     self.logger.info(f"  LOGGER- Mirroring UNet architecture with {len(down_block_types)} down blocks.")
    #     self.logger.info(f"  LOGGER- Down block types: {down_block_types}")
    #     self.logger.info(f"  LOGGER- Block out channels: {block_out_channels}")
    #     self.logger.info(f"  LOGGER- Setting use_linear_projection: {use_linear_projection}")
    #     self.logger.info(f"  LOGGER- cross_attention_dim: {cross_attention_dim}")
    #     self.logger.info(f"  LOGGER- transformer_layers_per_block: {transformer_layers_per_block}")
    #     self.logger.info(f"  LOGGER- projection_class_embeddings_input_dim: {projection_class_embeddings_input_dim}")

    #     # 2. Dynamically derive layer-specific parameters.
    #     # The 'attention_head_dim' in diffusers can be an int (dim) or a tuple (num_heads).
    #     # We need to provide what ControlNet expects.
    #     if isinstance(nnet_config.attention_head_dim, int):
    #         attention_head_dim = nnet_config.attention_head_dim
    #         num_attention_heads = tuple(c // attention_head_dim for c in block_out_channels)
    #     else:
    #         # If it's a tuple, it already represents num_attention_heads.
    #         # We must infer the head_dim. A common default is 64.
    #         num_attention_heads = nnet_config.attention_head_dim
    #         attention_head_dim = 64 # Assuming a standard head dimension
    #         self.logger.warning(f"  LOGGER- UNet config 'attention_head_dim' is a tuple. Using it as 'num_attention_heads': {num_attention_heads}")
    #         self.logger.warning(f"  LOGGER- Assuming 'attention_head_dim' is {attention_head_dim}.")

    #     self.logger.info(f"  LOGGER- Calculated num_attention_heads: {num_attention_heads}")
    #     self.logger.info(f"  LOGGER- Using attention_head_dim: {attention_head_dim}")

    #     # 3. Get other general parameters.
    #     num_control_type = getattr(self.config, 'num_control_type', 6)
    #     addition_time_embed_dim = getattr(nnet_config, 'addition_time_embed_dim', 256) or 256

    #     # The conditioning_embedding_out_channels must also match the number of blocks.
    #     # We'll truncate the default if the UNet has fewer blocks.
    #     default_cond_out_channels = (16, 32, 96, 256)
    #     conditioning_embedding_out_channels = default_cond_out_channels[:len(block_out_channels)+1]
    #     self.logger.info(f"  LOGGER- Adjusted conditioning_embedding_out_channels: {conditioning_embedding_out_channels}")


    #     controlnet = self.controlnet_class(
    #         # --- Mirrored Architecture from UNet ---
    #         addition_embed_type=addition_embed_type,
    #         down_block_types=down_block_types,
    #         block_out_channels=block_out_channels,
    #         layers_per_block=layers_per_block,
    #         transformer_layers_per_block=transformer_layers_per_block,
    #         attention_head_dim=attention_head_dim,
    #         num_attention_heads=num_attention_heads,
    #         cross_attention_dim=cross_attention_dim,
    #         projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,

    #         # --- Explicitly Corrected Parameter ---
    #         use_linear_projection=use_linear_projection,

    #         # --- Dynamic/Derived General Params ---
    #         in_channels=nnet_config.in_channels,
    #         conditioning_channels=3,
    #         num_control_type=num_control_type,
    #         addition_time_embed_dim=addition_time_embed_dim,

    #         # --- Other general params copied for compatibility ---
    #         flip_sin_to_cos=nnet_config.flip_sin_to_cos,
    #         freq_shift=nnet_config.freq_shift,
    #         downsample_padding=nnet_config.downsample_padding,
    #         mid_block_scale_factor=nnet_config.mid_block_scale_factor,
    #         act_fn=nnet_config.act_fn,
    #         norm_num_groups=nnet_config.norm_num_groups,
    #         norm_eps=nnet_config.norm_eps,
    #         upcast_attention=nnet_config.upcast_attention,
    #         resnet_time_scale_shift=nnet_config.resnet_time_scale_shift,

    #         # --- Dynamically Sized Params ---
    #         conditioning_embedding_out_channels=conditioning_embedding_out_channels,

    #         # --- Params with defaults ---
    #         global_pool_conditions=False,
    #     )
    #     return controlnet

    def build_controlnet(self):
        # Build a ControlNet that mirrors the UNet's architecture but with corrected parameters.
        self.logger.info("  LOGGER--- Calling build_controlnet ---")

        nnet_config = self.nnet.config

        self.logger.info("  LOGGER- Mirroring base UNet architecture...")
        
        use_linear_projection = nnet_config.use_linear_projection
        self.logger.info(f"  LOGGER- Adopting 'use_linear_projection' from UNet: {use_linear_projection}")


        attention_head_dim = nnet_config.attention_head_dim 
        num_attention_heads = None
        self.logger.info(f"  LOGGER- Aligning with legacy attention config:")
        self.logger.info(f"  LOGGER-   - Setting 'attention_head_dim' to list: {attention_head_dim}")
        self.logger.info(f"  LOGGER-   - Setting 'num_attention_heads' to: {num_attention_heads}")


        down_block_types = nnet_config.down_block_types
        block_out_channels = nnet_config.block_out_channels
        layers_per_block = nnet_config.layers_per_block
        cross_attention_dim = nnet_config.cross_attention_dim
        transformer_layers_per_block = nnet_config.transformer_layers_per_block
        
        addition_embed_type = nnet_config.addition_embed_type
        projection_class_embeddings_input_dim = nnet_config.projection_class_embeddings_input_dim
        addition_time_embed_dim = nnet_config.addition_time_embed_dim
        
        num_control_type = getattr(self.config, 'num_control_type', 6) 

        default_cond_out_channels = (16, 32, 96, 256)
        conditioning_embedding_out_channels = default_cond_out_channels[:len(block_out_channels)+1]


        controlnet = self.controlnet_class(
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            
            use_linear_projection=use_linear_projection,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,

            addition_embed_type=addition_embed_type,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            addition_time_embed_dim=addition_time_embed_dim,

            num_control_type=num_control_type,

            in_channels=nnet_config.in_channels,
            conditioning_channels=3,
            flip_sin_to_cos=nnet_config.flip_sin_to_cos,
            freq_shift=nnet_config.freq_shift,
            downsample_padding=nnet_config.downsample_padding,
            mid_block_scale_factor=nnet_config.mid_block_scale_factor,
            act_fn=nnet_config.act_fn,
            norm_num_groups=nnet_config.norm_num_groups,
            norm_eps=nnet_config.norm_eps,
            resnet_time_scale_shift=nnet_config.resnet_time_scale_shift,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            global_pool_conditions=False,
        )
        
        return controlnet


    def train_step(self, batch):
        # Prepare latents
        if batch.get('latents') is not None:
            latents = batch['latents'].to(self.device)
        else:
            # Ensure VAE parameters are in the expected dtype to avoid conv/bias dtype mismatch.
            try:
                vae_param_dtype = next(self.vae.parameters()).dtype
            except StopIteration:
                vae_param_dtype = None

            if vae_param_dtype is not None and vae_param_dtype != self.vae_dtype:
                # move model params to desired dtype on the device
                self.logger.info(f"  LOGGER- Moving VAE params from {vae_param_dtype} to {self.vae_dtype}")
                self.vae.to(device=self.device, dtype=self.vae_dtype)

            with torch.no_grad():
                latents = self.vae.encode(batch['images'].to(device=self.device, dtype=self.vae_dtype)).latent_dist.sample().to(self.weight_dtype)
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

    # Retrieve multi condition list & control type vector. This logic is now strict.
        union_control_type = batch.get('union_control_type')
        if union_control_type is None:
            raise ValueError(
                "Batch is missing 'union_control_type'. "
                "Please ensure your Dataset's __getitem__ method correctly returns this key."
            )
        
        condition_images_list = batch.get('condition_images_list')
        if condition_images_list is None:
            # For backward compatibility, we can check for the old 'condition_images' key.
            # If you want to be even stricter, you can remove this block.
            if batch.get('condition_images') is not None:
                self.logger.warning(
                    "Batch is using legacy 'condition_images' key. "
                    "Creating a 'condition_images_list' for backward compatibility. "
                    "Please update your dataset to provide 'condition_images_list' directly."
                )
                img = batch['condition_images'].to(self.device, dtype=self.controlnet.dtype)
                num_control_types = union_control_type.shape[1]
                condition_images_list = [img] + [torch.zeros_like(img) for _ in range(num_control_types - 1)]
            else:
                raise ValueError(
                    "Batch is missing 'condition_images_list'. "
                    "Please ensure your Dataset's __getitem__ method correctly returns this key."
                )

        # ---- Logging: 当前step的条件类型信息 ----
        # try:
        #     # union_control_type: [B, num_types]
        #     _uct = union_control_type.detach().cpu()
        #     # 简要统计每种control在batch中的激活数量
        #     _per_type_active = _uct.sum(dim=0).tolist()

        #     # condition_images_list: list(len = num_types) of [B, C, H, W]
        #     _ci_shapes = []
        #     for idx, ci in enumerate(condition_images_list):
        #         if ci is None:
        #             _ci_shapes.append(f"{idx}: None")
        #         else:
        #             shape = tuple(ci.shape)
        #             _ci_shapes.append(f"{idx}: {shape}")

        #     self.logger.debug(
        #         "[UnionTrain] union_control_type per-type active count: %s | condition_images_list shapes: %s",
        #         _per_type_active,
        #         "; ".join(_ci_shapes),
        #     )
        # except Exception as _log_e:
        #     self.logger.warning(f"[UnionTrain] failed to log union control info: {_log_e}")

        # --- Data processing and validation ---
        union_control_type = union_control_type.to(self.device)
        if union_control_type.dim() == 1:
            union_control_type = union_control_type.unsqueeze(0)
        if union_control_type.shape[0] != batch['images'].shape[0]:
            union_control_type = union_control_type.repeat(batch['images'].shape[0], 1)

        proc_list = []
        for idx, ci in enumerate(condition_images_list):
            if ci is None:
                # Allow None for inactive control types
                proc_list.append(None)
                continue
            if ci.shape[1] != 3:
                raise ValueError(
                    "SDXLControlNetUnionTrainer requires raw 3-channel images in 'condition_images_list'. "
                    f"Got {ci.shape[1]} channels. If you have pre-encoded features, change the dataset to return raw images."
                )
            proc_list.append(ci.to(self.device, dtype=self.controlnet.dtype))

        # ControlNet Union forward
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents.to(device=self.device, dtype=(self.controlnet.conv_in.weight.dtype if hasattr(self.controlnet.conv_in, 'weight') else noisy_latents.dtype)),
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
