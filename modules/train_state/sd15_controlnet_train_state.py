import os
import re
from waifuset import logging
from typing import List, Union
from safetensors.torch import save_file
from .sd15_train_state import SD15TrainState
from ..utils import eval_utils


class SD15ControlNetTrainState(SD15TrainState):
    save_full_model: bool = False
    nnet_trainable_params: List[Union[str, re.Pattern]] = None

    def save_controlnet_model(self):
        self.logger.print(f"saving controlnet model at epoch {self.epoch}, step {self.global_step}...")
        save_path = os.path.join(self.output_model_dir, f"{self.output_name['models']}_controlnet_ep{self.epoch}_step{self.global_step}.safetensors")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # self.unwrap_model(self.models.controlnet).save_pretrained(save_path, is_main_process=self.accelerator.is_main_process)
        controlnet = self.unwrap_model(self.controlnet)
        if self.accelerator.is_main_process:
            controlnet_sd = controlnet.state_dict()

            # print("--- DEBUG: Intercepted state_dict before saving ---")
            # print(f"Total keys in original state_dict: {len(controlnet_sd)}")
            
            # embedding_keys_found = [k for k in controlnet_sd.keys() if "controlnet_cond_embedding" in k]
            
            # if embedding_keys_found:
            #     print(">>> SUCCESS: 'controlnet_cond_embedding' keys ARE PRESENT in the state_dict before saving!")
            #     print("Keys found:", embedding_keys_found)
            # else:
            #     print(">>> CRITICAL: 'controlnet_cond_embedding' keys ARE MISSING from the state_dict even before saving!")
            #     print("This means the problem is with the 'self.controlnet' object itself.")

            # # (可选) 您甚至可以把所有键都打印出来看看
            # with open("debug_all_keys.txt", "w") as f:
            #     for key in sorted(controlnet_sd.keys()):
            #         f.write(f"{key}\n")
            # print("All keys have been written to debug_all_keys.txt")
            
            # print("--- DEBUG: End of interception ---")
            
            save_file(controlnet_sd, save_path)
        self.logger.print(f"controlnet model saved to: `{logging.yellow(save_path)}`")
        return save_path

    def save_diffusion_model(self):
        return None

    def get_pipeline(self):
        return self.pipeline_class(
            unet=self.unwrap_model(self.nnet),
            text_encoder=self.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            controlnet=self.unwrap_model(self.controlnet),
            vae=self.unwrap_model(self.vae),
            scheduler=eval_utils.get_sampler(self.eval_sampler),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            # clip_skip=self.clip_skip,
        )
