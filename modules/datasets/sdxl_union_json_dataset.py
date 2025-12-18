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


    def make_buckets(self, dataset: Dict[str, Any]) -> Dict[tuple, List[str]]:
        """
        覆写分桶逻辑：
        1. 调用父类方法，获取基于分辨率 (w, h) 的分桶。
        2. 将每个分辨率桶进一步按 'activated_modes' 拆分。
        这样生成的 Batch 既满足分辨率一致，也满足控制类型组合一致。
        """
        # 1. 获取父类的基础分桶 (已处理分辨率和权重)
        # resolution_buckets 结构: { (w, h): [img_key1, img_key1, img_key2...], ... }
        resolution_buckets = super().make_buckets(dataset)
        
        final_buckets = {}
        
        for resolution, img_keys in resolution_buckets.items():
            # 临时字典：在当前分辨率下，按控制模式分组
            mode_groups: Dict[tuple, List[str]] = {}
            
            for key in img_keys:
                img_md = dataset[key]
                # 获取该样本的控制模式，转为 tuple 以便作为字典 key
                # 如果没有 modes 字段，默认为空 tuple
                modes = tuple(img_md.get('activated_modes', []))
                
                if modes not in mode_groups:
                    mode_groups[modes] = []
                mode_groups[modes].append(key)
            
            # 将拆分后的组放回最终 buckets
            # 新的 Key 结构: (w, h, mode1, mode2, ...)
            # 这样既保留了 w, h 在前两位（兼容父类排序逻辑），又区分了控制类型
            for modes, keys in mode_groups.items():
                new_key = resolution + modes
                final_buckets[new_key] = keys

        self.logger.info(f"--- Union Bucket Statistics ---")
        self.logger.info(f"Total Buckets: {len(final_buckets)}")
        total_remainder_batches = 0
        bs = getattr(self, 'batch_size', 8) 
        print(f"Using batch size: {bs}")
        for b_key, b_imgs in final_buckets.items():
            count = len(b_imgs)
            # 假设 batch_size 可以从 self.batch_size 获取，或者硬编码 8 用于观察
            # 注意：dataset 实例通常有 batch_size 属性，如果没有，这行仅作打印参考
            
            remainder = count % bs
            if remainder != 0:
                self.logger.info(f"Bucket {b_key} has {count} images. Remainder: {remainder} (Will create a partial batch of size {remainder})")
                total_remainder_batches += 1
        
        if total_remainder_batches == 0:
            self.logger.info("Perfect! All buckets are divisible by batch size.")
        else:
            self.logger.info(f"Found {total_remainder_batches} buckets that will produce a partial batch.")
        
        return final_buckets

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
            file_path = item['file'] # 例如: "00000-0-100086946_p0/image.png"
            modes = item.get('modes', [])
            caption_text = item.get('caption') or ''
            
            if not modes:
                continue
            
            # --- 核心修复逻辑开始 ---
            # 1. 获取文件名和父目录名
            dir_name = os.path.dirname(file_path) # "00000-0-100086946_p0"
            base_name = os.path.basename(file_path) # "image.png"
            file_stem_name = os.path.splitext(base_name)[0] # "image"

            # 2. 智能判断 ID
            if file_stem_name == 'image' and dir_name:
                file_stem = dir_name
            else:
                # 否则可能是扁平结构 (ID.png)，保持原有逻辑
                file_stem = file_stem_name
            # --- 核心修复逻辑结束 ---
            
            img_fp = os.path.join(self.images_root, file_path)
            
            # 简单检查原图是否存在，不存在则跳过，避免无效数据
            if not os.path.exists(img_fp):
                # self.logger.warning(f"Image not found: {img_fp}") 
                continue
            
            for mode_list in modes:
                # Validation
                for m in mode_list:
                    if m not in self.control_type_order:
                        raise ValueError(f"Control type `{m}` not in order list.")
                
                entry_key = f"{file_stem}__{sample_index}"
                expanded[entry_key] = {
                    'image_key': entry_key,
                    'image_path': img_fp,
                    'file_stem': file_stem, # 现在这是正确的 ID 了 (例如 00000..._p0)
                    'activated_modes': mode_list, 
                    'caption': caption_text,
                    # 缓存一些不需要重复计算的路径元数据，方便 _load_pure_pil_control 使用
                    # 'raw_file_path': file_path 
                }
                sample_index += 1
        
        self.dataset = expanded
        self.logger.info(f"Loaded {len(self.dataset)} Union samples. Types: {self.control_type_order}")


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

        if not self.generate_missing_controls:
            self.logger.warning(
                f"[Dataset Debug] Control '{control_type}' missing for '{file_stem}'. "
                f"Searched paths: {candidates}"
            )

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
                        raise Exception(f"Control image for type '{control_name}' could not be loaded or generated for sample '{img_key}'.")
                    
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