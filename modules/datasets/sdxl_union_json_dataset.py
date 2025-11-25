import os
import json
import torch
from typing import List, Dict, Any
from PIL import Image

from .sdxl_dataset import SDXLDataset
from ..utils import dataset_utils

try:
    from controlnet_aux.processor import Processor
except ImportError:
    Processor = None



class SDXLUnionJSONDataset(SDXLDataset):
    """Dataset driven by a JSON manifest controlling per-image control modes.

    JSON schema (example):
    {
      "images_root": "/path/to/images",
      "controls_root": "/path/to/controls",         # optional root for precomputed control images
      "control_type_order": ["openpose","depth_midas","canny","lineart","normal","segment"],
      "entries": [
        {"file": "0001.jpg", "modes": [["canny"],["depth_midas"],["openpose"]]},
        {"file": "0002.jpg", "modes": [["canny","depth_midas"],["openpose"]]}
      ]
    }

    Expansion rules:
      - Each element in `modes` becomes one training sample (batch size=1 flow).
      - A mode list of length >1 is a composite (multi condition simultaneously).
    If an image is absent from JSON it is never loaded.
    """

    union_json_path: str = None
    generate_missing_controls: bool = False  # if True and precomputed control not found, attempt on-the-fly generation
    controls_root_fallback: str = None       # optional fallback root

    def check_config(self):
        super().check_config()
        if not self.union_json_path:
            raise ValueError("`union_json_path` must be provided for SDXLUnionJSONDataset")
        if not os.path.exists(self.union_json_path):
            raise FileNotFoundError(f"Union JSON not found: {self.union_json_path}")

    def _setup_dataset(self):
        # Override base loading: we load from JSON only
        with open(self.union_json_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        self.images_root = manifest.get('images_root') or ''
        self.controls_root = manifest.get('controls_root') or ''
        self.control_type_order: List[str] = manifest.get('control_type_order') or []
        entries: List[Dict[str, Any]] = manifest.get('entries') or []
        if not self.control_type_order:
            raise ValueError("control_type_order must be specified in JSON")
        # Build expanded dataset list.
        expanded = {}
        sample_index = 0
        for item in entries:
            file = item['file']
            modes = item.get('modes', [])
            if not modes:
                continue
            img_fp = os.path.join(self.images_root, file)
            if not os.path.exists(img_fp):
                self.logger.warning(f"Image missing: {img_fp}, skip")
                continue
            for mode_list in modes:  # each mode_list is a list of one or multiple control types
                # validate modes
                for m in mode_list:
                    if m not in self.control_type_order:
                        raise ValueError(f"Control type `{m}` not in control_type_order: {self.control_type_order}")
                entry_key = f"{file}__{sample_index}"
                expanded[entry_key] = {
                    'image_key': entry_key,
                    'image_path': img_fp,
                    'file': file,
                    'activated_modes': mode_list,
                }
                sample_index += 1
        self.dataset = expanded  

    def get_basic_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict:
        # Reuse original size logic from parent but inject activated modes
        base_sample = super().get_basic_sample(batch, samples)
        # For each single entry (batch size=1 typical), attach union control info
        activated_modes_batch = []
        for img_key in batch:
            img_md = self.dataset[img_key]
            activated_modes_batch.append(img_md['activated_modes'])
        base_sample['activated_modes'] = activated_modes_batch
        return base_sample

    def get_condition_images_list_and_union_vector(self, img_md, activated_modes, image_tensor):
        """Construct (condition_images_list, union_control_type) for a single sample.
        image_tensor is the original image (B=1) to allow fallback generation.
        Returns tensors ready for stacking later.
        """
        num_types = len(self.control_type_order)
        union_vec = torch.zeros(num_types, dtype=torch.float32)
        cond_list = []
        # Determine reference H/W from image_tensor
        if image_tensor is None:
            raise ValueError("image_tensor required for control construction")
        _, _, H, W = image_tensor.shape
        # Build map
        for idx, ctype in enumerate(self.control_type_order):
            if ctype in activated_modes:
                union_vec[idx] = 1.0
                cond_img = self.load_or_generate_control(img_md, ctype, (W, H))
            else:
                cond_img = torch.zeros(3, H, W)
            cond_list.append(cond_img)
        cond_stack = torch.stack(cond_list, dim=0)  # (num_types,3,H,W)
        return cond_stack, union_vec

    def load_or_generate_control(self, img_md, ctype: str, target_size):
        """Load precomputed control image or optionally generate on-the-fly using controlnet_aux Processor."""
        # Precomputed path convention: controls_root/<ctype>/<stem>.png
        if self.controls_root:
            stem = os.path.splitext(img_md['file'])[0]
            fp = os.path.join(self.controls_root, ctype, f"{stem}.png")
            if os.path.exists(fp):
                image = Image.open(fp).convert('RGB')
                image = dataset_utils.resize_if_needed(image, target_size, resampling='lanczos')
                return dataset_utils.IMAGE_TRANSFORMS(image)
        if self.generate_missing_controls:
            if Processor is None:
                raise ImportError("controlnet_aux not installed; cannot generate controls on-the-fly")
            proc = Processor(ctype)
            pil_img = self.open_image(img_md)
            pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
            ctrl = proc(pil_img, to_pil=True)
            ctrl = dataset_utils.resize_if_needed(ctrl, target_size, resampling='lanczos')
            return dataset_utils.IMAGE_TRANSFORMS(ctrl)
        # Fallback: zeros (will still set union_vec bit so model learns position bias?)
        return torch.zeros(3, target_size[1], target_size[0])

    def collate_fn(self, batch: List[str]) -> Dict[str, Any]:
        # Build samples with parent samplers
        samples: Dict[str, Any] = {}
        for sampler in self.get_samplers():
            samples.update(sampler(batch, samples))
        # Expect batch size=1 typical; but handle >1 generically
        condition_images_lists = []
        union_vectors = []
        for b_idx in range(len(batch)):
            img_key = batch[b_idx]
            img_md = self.dataset[img_key]
            activated_modes = img_md['activated_modes']
            # image tensor shape (B,3,H,W); we take single
            image_tensor = samples['images'][b_idx].unsqueeze(0) if samples['images'] is not None else samples['latents'][b_idx].unsqueeze(0)
            cond_stack, union_vec = self.get_condition_images_list_and_union_vector(img_md, activated_modes, image_tensor)
            condition_images_lists.append(cond_stack)  # (num_types,3,H,W)
            union_vectors.append(union_vec)
        # Stack along batch dim: we keep trainer expectations: condition_images_list -> list per control type? We instead provide tensor per sample.
        # Trainer will adapt: provide condition_images_list as list[tensor] length=num_types
        # Transform shape (B,num_types,3,H,W) -> list[num_types] each (B,3,H,W)
        cond_tensor = torch.stack(condition_images_lists, dim=0)  # B,num_types,3,H,W
        cond_per_type = [cond_tensor[:, i] for i in range(cond_tensor.shape[1])]  # list length num_types
        union_control_type = torch.stack(union_vectors, dim=0)  # B,num_types
        samples['condition_images_list'] = cond_per_type
        samples['union_control_type'] = union_control_type
        return samples
