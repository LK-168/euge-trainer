import os
import json
import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Literal, Callable
from PIL import Image
from torchvision import transforms

# 引入父类和工具
from .sdxl_dataset import SDXLDataset
from ..utils import dataset_utils
from waifuset import logging

# 尝试引入 Processor
try:
    from controlnet_aux.processor import Processor
except ImportError:
    Processor = None

CNAUX_PROCESSORS = {}

# 这里的 Transform 和你提供的模板保持一致
CONDITION_IMAGE_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
])

def get_controlnet_aux_condition(
    image: Image.Image,
    condition_type: str,
    **kwargs
) -> Image.Image:
    r"""
    Get the condition of an image using controlnet_aux library.
    Shared helper function.
    """
    global CNAUX_PROCESSORS
    if condition_type not in CNAUX_PROCESSORS:
        if Processor is None:
            raise ImportError("controlnet_aux is not installed.")
        # Lazy load processor
        CNAUX_PROCESSORS[condition_type] = Processor(condition_type, params=kwargs)
    
    processor = CNAUX_PROCESSORS[condition_type]
    condition = processor(image, to_pil=True)
    
    # Resize consistency check
    target_width, target_height = image.size
    if condition.width != target_width or condition.height != target_height:
        condition = condition.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
    
    return condition


class SDXLUnionJSONDataset(SDXLDataset):
    """Dataset driven by a JSON manifest controlling per-image control modes.

    JSON schema (example):
    {
      "images_root": "/path/to/images",
      "controls_root": "/path/to/controls",         # root for precomputed control images
      "control_type_order": ["openpose","depth_midas","canny","lineart","normal","segment"],
      "entries": [
        {"file": "0001.jpg", "modes": [["canny"],["depth_midas"],["openpose"]]},
        {"file": "0002.jpg", "modes": [["canny","depth_midas"],["openpose"]]}
      ]
    }
    """

    # Configurable attributes
    union_json_path: str = None
    generate_missing_controls: bool = False
    controls_root_fallback: str = None
    
    # Visualisation/Caching settings (inherited concept from ImageConditionDataset)
    condition_image_resampling: str = 'lanczos'

    def check_config(self):
        super().check_config()
        if not self.union_json_path:
            raise ValueError("`union_json_path` must be provided for SDXLUnionJSONDataset")
        if not os.path.exists(self.union_json_path):
            raise FileNotFoundError(f"Union JSON not found: {self.union_json_path}")
        
        if self.generate_missing_controls and Processor is None:
            self.logger.warning("`generate_missing_controls` is True but `controlnet_aux` is not installed.")

    def _setup_dataset(self):
        self.logger.info(f"Loading Union JSON from {self.union_json_path}...")
        with open(self.union_json_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        self.images_root = manifest.get('images_root') or ''
        self.controls_root = manifest.get('controls_root') or self.controls_root_fallback or ''
        self.control_type_order: List[str] = manifest.get('control_type_order') or []
        entries: List[Dict[str, Any]] = manifest.get('entries') or []

        if not self.control_type_order:
            raise ValueError("control_type_order must be specified in JSON.")

        self.num_control_types = len(self.control_type_order)
        
        # Build dataset
        expanded = {}
        sample_index = 0
        
        for item in entries:
            file = item['file']
            modes = item.get('modes', [])
            caption_text = item.get('caption') or ''
            
            if not modes:
                continue
            
            img_fp = os.path.join(self.images_root, file)
            if not os.path.exists(img_fp):
                continue
            
            file_stem = os.path.splitext(os.path.basename(file))[0]

            for mode_list in modes:
                # Validation
                for m in mode_list:
                    if m not in self.control_type_order:
                        raise ValueError(f"Control type `{m}` not in order list.")
                
                entry_key = f"{file_stem}__{sample_index}"
                expanded[entry_key] = {
                    'image_key': entry_key,
                    'image_path': img_fp,
                    'file_stem': file_stem,
                    'activated_modes': mode_list, # List of active controls for this sample
                    'caption': caption_text,
                }
                sample_index += 1
        
        self.dataset = expanded
        self.logger.info(f"Loaded {len(self.dataset)} Union samples. Types: {self.control_type_order}")

    # def get_samplers(self):
    #     return [
    #         self.get_basic_sample,
    #         self.get_size_sample,
    #         self.get_control_sample,
    #     ]

    # def get_samplers(self):
    #     samplers = super().get_samplers()

    #     control_sampler = getattr(self, "get_control_sample", None)
    #     if control_sampler is None:
    #         return samplers

    #     if control_sampler in samplers:
    #         samplers.remove(control_sampler)

    #     insert_idx = len(samplers)

    #     for idx, func in enumerate(samplers):
    #         name = getattr(func, "__name__", "")
    #         if name.startswith("get_control_") and func is not control_sampler:
    #             insert_idx = idx
    #             break

    #     samplers.insert(insert_idx, control_sampler)
    #     return samplers

    def _load_pure_pil_control(self, img_md: Dict, control_type: str) -> Optional[Image.Image]:
        """Loads a raw PIL control image from disk or generates it, without resizing/cropping."""
        file_stem = img_md['file_stem']
        
        # 1. Try Load from Disk
        candidates = [
            os.path.join(self.controls_root, control_type, f"{file_stem}.png"),
            os.path.join(self.controls_root, control_type, f"{file_stem}.jpg"),
            os.path.join(self.controls_root, f"{file_stem}_{control_type}.png"),
        ]
        
        for p in candidates:
            if os.path.exists(p):
                try:
                    return Image.open(p).convert("RGB")
                except Exception as e:
                    self.logger.warning(f"Failed to open {p}: {e}")

        # 2. Try Generate
        if self.generate_missing_controls:
            # We need the original image to generate
            original_pil = self.open_image(img_md) 
            if original_pil is not None:
                try:
                    return get_controlnet_aux_condition(original_pil, control_type)
                except Exception as e:
                    self.logger.warning(f"Failed to generate {control_type} for {img_md['image_key']}: {e}")
        
        return None


    def get_size_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict:
        if 'is_flipped' not in samples:
            basic_out = self.get_basic_sample(batch, samples)
            samples.update(basic_out)
        
        return super().get_size_sample(batch, samples)

    def get_condition_image(self, img_md, type: Literal['pil', 'tensor', 'numpy'] = 'tensor') -> Any:
        """
        Retrieves a representative condition image for visualization/validation.
        For Union dataset, this picks the FIRST activated mode in the list.
        """
        activated_modes = img_md.get('activated_modes', [])
        
        # If no modes active (unlikely), or fails to load, return None or Blank
        condition_image = None
        
        if activated_modes:
            # For visualization, we just pick the first one
            primary_mode = activated_modes[0]
            condition_image = self._load_pure_pil_control(img_md, primary_mode)
        
        # Fallback if load failed
        if condition_image is None:
            # Use a black image of original size so code doesn't crash
            _, orig_size, _ = self.get_size(img_md, update=False)
            condition_image = Image.new("RGB", orig_size, (0, 0, 0))

        # --- Apply Transforms (Resize -> Crop -> Resize to Bucket) ---
        # This mirrors dataset_utils logic
        image_size, _, bucket_size = self.get_size(img_md, update=True)
        crop_ltrb = self.get_crop_ltrb(img_md, update=True)

        condition_image = dataset_utils.resize_if_needed(condition_image, image_size, resampling=self.condition_image_resampling)
        condition_image = dataset_utils.crop_ltrb_if_needed(condition_image, crop_ltrb)
        condition_image = dataset_utils.resize_if_needed(condition_image, bucket_size, resampling=self.condition_image_resampling)

        # Return in requested format
        if type == 'tensor':
            return CONDITION_IMAGE_TRANSFORMS(condition_image)
        elif type == 'numpy':
            return np.array(condition_image)
        elif type == 'pil':
            return condition_image
        else:
            raise ValueError(f"Invalid type: {type}")

    # =========================================================================
    # Method 2: Required by Trainer Training Loop (The Union Logic)
    # =========================================================================

    def get_control_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict:
        """
        Builds 'condition_images_list' and 'union_control_type' for the batch.
        """
        batch_size = len(batch)
        
        # Prepare containers
        # list of [B, C, H, W] tensors
        batch_conditions_per_type = [[] for _ in range(self.num_control_types)]
        # [B, N_types]
        union_control_type_tensor = torch.zeros((batch_size, self.num_control_types), dtype=torch.float32)

        for i, img_key in enumerate(batch):
            img_md = self.dataset[img_key]
            activated_modes = img_md['activated_modes']
            
            # Geometry info from previous samplers
            bucket_size = img_md['bucket_size'] # target (w, h)
            crop_ltrb = img_md['crop_ltrb']
            # is_flipped = samples['is_flipped'][i]

            is_flipped_list = samples.get('is_flipped')
            if is_flipped_list is None:
                is_flipped = False
            else:
                is_flipped = is_flipped_list[i]

            # Iterate all global types
            for type_idx, control_name in enumerate(self.control_type_order):
                is_active = control_name in activated_modes
                
                ctrl_tensor = None
                
                if is_active:
                    # 1. Load Raw
                    ctrl_img = self._load_pure_pil_control(img_md, control_name)
                    
                    if ctrl_img is None:
                        # Fallback: black image
                        ctrl_img = Image.new("RGB", img_md['image_size'], (0,0,0))
                    
                    # 2. Mark vector
                    union_control_type_tensor[i, type_idx] = 1.0

                    # 3. Geometric Transforms (Must match T2I logic perfectly)
                    # Resize to working resolution
                    ctrl_img = dataset_utils.resize_if_needed(ctrl_img, img_md['image_size'], resampling=self.condition_image_resampling)
                    # Crop
                    ctrl_img = dataset_utils.crop_ltrb_if_needed(ctrl_img, crop_ltrb)
                    # Resize to Bucket
                    ctrl_img = dataset_utils.resize_if_needed(ctrl_img, bucket_size, resampling=self.condition_image_resampling)
                    
                    # 4. Flip
                    if is_flipped:
                        ctrl_img = ctrl_img.transpose(Image.FLIP_LEFT_RIGHT)

                    # 5. To Tensor
                    ctrl_tensor = CONDITION_IMAGE_TRANSFORMS(ctrl_img)
                
                else:
                    # Inactive: Zero tensor of shape [3, H, W]
                    # bucket_size is (W, H), tensor needs (H, W)
                    h, w = bucket_size[1], bucket_size[0]
                    ctrl_tensor = torch.zeros((3, h, w), dtype=torch.float32)

                batch_conditions_per_type[type_idx].append(ctrl_tensor)

        # Stack into batch tensors
        final_list = []
        for type_idx in range(self.num_control_types):
            stack = torch.stack(batch_conditions_per_type[type_idx], dim=0)
            final_list.append(stack)

        samples['condition_images_list'] = final_list
        samples['union_control_type'] = union_control_type_tensor

        return samples