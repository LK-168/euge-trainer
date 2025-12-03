import io
import torch
from PIL import Image
from typing import List, Dict, Any

from .sdxl_image_condition_dataset import SDXLImageConditionDataset
from ..utils import dataset_utils

class SDXLUnionParquetDataset(SDXLImageConditionDataset):
    """
    A dataset that loads images and multiple pre-computed control maps directly from a Parquet file.
    The Parquet file is expected to contain columns for the image bytes and each control type's bytes.
    Which control types to use for a given sample is determined by `control_type_order` and `active_control_modes` in the config.
    """

    def check_config(self):
        super().check_config()
        # Default order based on user-provided parquet schema
        default_order = [
            "canny",
            "depth_midas",
            "dwpose",
            "lineart_anime",
            "lineart_realistic",
            "manga_line",
            "scribble_hed",
            "scribble_pidinet",
        ]
        self.control_type_order = self.config.get('control_type_order', default_order)
        self.active_control_modes = self.config.get('active_control_modes', [])
        if not self.control_type_order:
            raise ValueError("`control_type_order` must be defined or inferable for SDXLUnionParquetDataset.")
        # When active_control_modes is empty, we won't expand combinations; we'll use per-row available modes
        if not self.active_control_modes:
            self.logger.info("`active_control_modes` not provided; each sample will use only its available control types (no combinations).")

    def _setup_dataset(self):
        # The parent class T2IDataset already loads parquet files into self.dataset using waifuset.
        super()._setup_dataset()

        original_dataset = list(self.dataset.values())
        expanded_dataset = {}
        sample_idx = 0

        if not original_dataset:
            self.logger.warning("Parquet dataset appears empty after setup.")
        
        for img_md in original_dataset:
            # Determine available modes per row (bytes/PIL present)
            available_modes = []
            for mode in self.control_type_order:
                if mode in img_md and img_md[mode] is not None:
                    val = img_md[mode]
                    if isinstance(val, (bytes, Image.Image)):
                        available_modes.append(mode)
            # If explicit combinations provided, expand; otherwise one sample per row with available modes only
            if self.active_control_modes:
                for modes in self.active_control_modes:
                    # Validate all requested modes exist in this row
                    if all(m in available_modes for m in modes):
                        new_key = f"{img_md.get('image_key', 'row')}_{sample_idx}"
                        new_entry = img_md.copy()
                        new_entry['image_key'] = new_key
                        new_entry['activated_modes'] = modes
                        expanded_dataset[new_key] = new_entry
                        sample_idx += 1
            else:
                # No combinations: just use per-row available modes
                if available_modes:
                    new_key = f"{img_md.get('image_key', 'row')}_{sample_idx}"
                    new_entry = img_md.copy()
                    new_entry['image_key'] = new_key
                    new_entry['activated_modes'] = available_modes
                    expanded_dataset[new_key] = new_entry
                    sample_idx += 1

        self.dataset = expanded_dataset
        if self.active_control_modes:
            self.logger.info(f"Expanded dataset to {len(self.dataset)} samples using provided active_control_modes combinations.")
        else:
            self.logger.info(f"Prepared dataset with {len(self.dataset)} samples; each sample uses only its own available control types (no combinations).")

    def open_image(self, img_md: Dict[str, Any]) -> Image.Image:
        """Load image from parquet row supporting bytes or already-loaded PIL/Image types."""
        val = img_md.get("image")
        if isinstance(val, bytes):
            return Image.open(io.BytesIO(val)).convert("RGB")
        if isinstance(val, Image.Image):
            return val.convert("RGB")
        # Fallback to parent's logic if 'image' column is a path or other type
        return super().open_image(img_md)

    def open_condition_image(self, img_md: Dict[str, Any], control_type: str) -> Image.Image:
        """Open a condition image from parquet row supporting bytes or PIL/Image types."""
        val = img_md.get(control_type)
        if isinstance(val, bytes):
            return Image.open(io.BytesIO(val)).convert("RGB")
        if isinstance(val, Image.Image):
            return val.convert("RGB")
        # Silent None to allow zero-tensor fallback
        return None

    def get_sample_extra(self, batch: List[str], samples: Dict[str, Any]) -> Dict[str, Any]:
        """
        This sampler is responsible for creating the `condition_images_list` and `union_control_type`
        tensors required by the SDXLControlNetUnionTrainer.
        """
        condition_images_list_batch = []
        union_control_type_batch = []

        for img_key in batch:
            img_md = self.dataset[img_key]
            activated_modes = img_md.get('activated_modes', [])
            
            _, _, bucket_size = self.get_size(img_md)
            
            current_sample_cond_list = []
            union_vector = [1.0 if mode in activated_modes else 0.0 for mode in self.control_type_order]

            for control_type in self.control_type_order:
                if control_type in activated_modes:
                    # This image is active, load it
                    cond_image = self.open_condition_image(img_md, control_type)
                    if cond_image:
                        # We need to apply the same transforms as the main image (resize, crop)
                        # Reusing get_condition_image logic from parent but with our open method
                        image_size, _, _ = self.get_size(img_md)
                        crop_ltrb = self.get_crop_ltrb(img_md)
                        cond_image = dataset_utils.resize_if_needed(cond_image, image_size, resampling=self.condition_image_resampling)
                        cond_image = dataset_utils.crop_ltrb_if_needed(cond_image, crop_ltrb)
                        cond_image = dataset_utils.resize_if_needed(cond_image, bucket_size, resampling=self.condition_image_resampling)
                        cond_tensor = dataset_utils.CONDITION_IMAGE_TRANSFORMS(cond_image)
                    else: # Should not happen due to check in _setup_dataset
                        cond_tensor = torch.zeros(3, bucket_size[1], bucket_size[0])
                else:
                    # This image is not active, provide a zero tensor
                    cond_tensor = torch.zeros(3, bucket_size[1], bucket_size[0])
                
                current_sample_cond_list.append(cond_tensor)

            # The trainer expects a list of tensors, where each tensor is a batch for one control type.
            # Here we build it for a single sample, then will rearrange.
            condition_images_list_batch.append(torch.stack(current_sample_cond_list)) # Shape: [num_control_types, 3, H, W]
            union_control_type_batch.append(torch.tensor(union_vector, dtype=torch.float32))

        # Stack all samples in the batch
        # Shape: [B, num_control_types, 3, H, W]
        batch_cond_tensor = torch.stack(condition_images_list_batch)
        
        # Rearrange for the trainer: a list of length `num_control_types`,
        # where each element is a tensor of shape `[B, 3, H, W]`.
        final_condition_images_list = [batch_cond_tensor[:, i, :, :, :] for i in range(len(self.control_type_order))]

        samples.update({
            "condition_images_list": final_condition_images_list,
            "union_control_type": torch.stack(union_control_type_batch),
        })
        return samples

    def get_samplers(self) -> List:
        # Override to inject our custom sampler that provides the union-style tensors
        samplers = super().get_samplers()
        # Remove the default condition image sampler if it exists
        # It's named get_condition_image_sample in the parent class ImageConditionDataset
        if hasattr(self, 'get_condition_image_sample') and self.get_condition_image_sample in samplers:
            samplers.remove(self.get_condition_image_sample)
        samplers.append(self.get_sample_extra)
        return samplers
