from ml_collections import ConfigDict


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ConfigDict()
    config.pretrained_model_name_or_path = '/root/data-local/z_qwen/noobai-XL-1.1'
    config.output_dir = '/root/data-local/z_qwen/output/union_test2'
    config.vae_model_name_or_path = None
    config.hf_cache_dir = None

    # Dataset sources (image + captions). Provide your own path.
    config.dataset_source = [
        dict(name_or_path='/root/data-local/z_qwen/dataset/anicontrol-20k/processed_data', read_attrs=True),
    ]

    # Multi-condition parameters
    # Provide a list of condition types for dynamic sampling, e.g. ['canny','depth_midas','openpose']
    # config.union_condition_types = [
    #     'openpose',
    #     'depth_midas',
    #     'canny',
    #     'lineart_anime',
    #     'lineart_realistic',
    #     'manga_line',
    #     'scribble_hed',
    #     'scribble_pidinet',
    #     'dwpose',
    # ]
    # config.num_control_type = 9

    config.union_condition_types = [
        'depth_midas',
        'canny',
    ]
    config.num_control_type = len(config.union_condition_types)


    # Optional JSON manifest to drive per-image modes; if set, dataset class switches automatically
    # Example JSON schema documented in sdxl_union_json_dataset.py
    config.union_json_path = '/root/data-local/z_qwen/anicontrol-20k/union_manifest.json'
    config.generate_missing_controls = False  # if True, will attempt on-the-fly generation when file missing


    # Resolution / buckets
    # config.resolution = 1024
    config.resolution = 1024
    config.allow_crop = False
    config.random_crop = False
    config.arb = True
    config.flip_aug = True

    # Model / tokenizer
    config.no_half_vae = False
    config.use_deepspeed = False
    config.tokenizer_cache_dir = 'tokenizers'
    config.max_token_length = 225
    config.max_retries = None

    # Bucket settings
    config.max_width = None
    config.max_height = None
    config.max_area = 1024 * 1024
    config.bucket_reso_step = 32
    config.max_aspect_ratio = 1.1
    config.predefined_buckets = [
        (1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
        (1344, 768), (768, 1344), (1536, 640), (640, 1536), (1792, 576), (576, 1792), (2048, 512), (512, 2048)
    ]
    config.vae_batch_size = 6
    config.max_dataset_n_workers = 1
    config.max_dataloader_n_workers = 16
    config.persistent_data_loader_workers = False

    # Output organization
    config.output_subdir = cfg(models='models', train_state='train_state', samples='samples', logs='logs')
    config.output_name = cfg(models=None, train_state=None)
    config.loss_recorder_kwargs = cfg(
        gamma=0.9,
        stride=1000,
    )
    config.save_precision = 'fp16'
    config.save_model = True
    config.save_train_state = True
    config.save_every_n_epochs = 0
    config.save_every_n_steps = 1000
    config.save_on_train_start = False
    config.save_on_train_end = True
    config.save_on_keyboard_interrupt = True
    config.save_on_exception = True
    config.save_max_n_models = 5
    config.save_max_n_train_states = 2

    # Sampling / evaluation
    config.eval_train_size = 1
    config.eval_valid_size = 0
    config.eval_every_n_epochs = 0
    config.eval_every_n_steps = 1000
    config.eval_on_train_start = True
    config.eval_sampler = 'euler_a'
    config.eval_params = cfg(
        prompt='1girl, solo, detailed, high quality',
        negative_prompt='low quality, bad anatomy',
        steps=30,
        batch_size=1,
        batch_count=1,
        scale=7.0,
        seed=42,
        width=1024,
        height=1024,
        save_latents=False,
        control_scale=1.0,
    )

    # Training basic params
    config.num_train_epochs = 1
    config.batch_size = 1
    config.learning_rate = 1e-5
    config.train_nnet = False
    config.learning_rate_nnet = 1e-5
    config.train_controlnet = True
    config.learning_rate_controlnet = 1e-5
    config.lr_scheduler = 'constant_with_warmup'
    config.lr_warmup_steps = 500
    config.lr_scheduler_power = 1.0
    config.lr_scheduler_num_cycles = 1
    config.lr_scheduler_kwargs = cfg()
    config.mixed_precision = 'bf16'
    config.full_bf16 = True
    config.full_fp16 = False

    # config.mixed_precision = 'no'
    # config.full_bf16 = False
    # config.full_fp16 = False

    config.train_text_encoder = False
    config.learning_rate_te = 5e-6
    config.gradient_checkpointing = True
    config.gradient_accumulation_steps = 16
    config.optimizer_type = 'AdaFactor'
    
    config.optimizer_kwargs = cfg(
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        # weight_decay=0.03,
        # betas=(0.9, 0.9),
        # amsgrad=False
    )
    # config.optimizer_type = 'AdamW'
    # config.optimizer_kwargs = cfg(betas=(0.9,0.99), weight_decay=0.01)
    
    # config.optimizer_type = 'SGDNesterov'
    # config.optimizer_kwargs = cfg(momentum=0.9, weight_decay=0.01)

    # config.optimizer_type = 'AdamW8bit'
    # config.optimizer_kwargs = cfg(betas=(0.9,0.99), weight_decay=0.01)

    config.cpu = False
    # Advanced noise / guidance options
    config.mem_eff_attn = False
    config.xformers = True
    config.diffusers_xformers = False
    config.sdpa = False
    # config.mem_eff_attn = False
    # config.xformers = False
    # config.diffusers_xformers = False
    # config.sdpa = False    

    config.clip_skip = None
    config.noise_offset = 0
    config.multires_noise_iterations = 0
    config.multires_noise_discount = 0
    config.adaptive_noise_scale = None
    config.max_grad_norm = 1.0
    config.prediction_type = 'epsilon'
    config.zero_terminal_snr = False
    config.ip_noise_gamma = 0
    config.min_snr_gamma = 0
    config.debiased_estimation_loss = False
    config.min_timestep = 0
    config.max_timestep = 1000
    config.timestep_sampler_type = 'uniform'

    # Cache latents
    config.cache_latents = False
    config.cache_latents_to_disk = True
    config.cache_only = False
    config.check_cache_validity = False
    config.keep_cached_latents_in_memory = False
    config.async_cache = True
    config.cache_latents_max_dataloader_n_workers = 32

    return config
