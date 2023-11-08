#ref: https://github.com/M1kep/ComfyUI-OtherVAEs

from typing import Tuple
import torch

import folder_paths
from comfy import model_management
from comfy.taesd.taesd import TAESD

class TAESDDecoder:
    def __init__(self, file_name: str, max_batch_size: int) -> None:
        self.taesd = TAESD(None, folder_paths.get_full_path("vae_approx", file_name)).to(model_management.get_torch_device())
        self.scale = 0.13025 if "xl" in file_name else 0.18215 # This is a hack, but it works for now
        self.max_batch_size = max_batch_size

    @torch.no_grad()
    def decode(self, latent):
        B = latent.shape[0]
        latent = latent.to(model_management.get_torch_device()) * self.scale
        x_sample = []
        for i in range(0, B, self.max_batch_size):
            x_sample.append(self.taesd.decoder(latent[i:i + self.max_batch_size]).detach())
        x_sample = torch.cat(x_sample, dim=0)
        x_sample = x_sample.sub(0.5).mul(2)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = x_sample.permute(0, 2, 3, 1).cpu()
        return x_sample


class TAESDLoader:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "file_name": (folder_paths.get_filename_list("vae_approx"), {}),
                "max_batch_size": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    OUTPUT_IS_LIST = (False,)
    CATEGORY = "loaders"

    def __init__(self):
        self.taesd = None
        self.file_name = None

    def load(self, file_name: str, max_batch_size: int) -> Tuple[torch.Tensor]:
        if self.file_name != file_name:
            self.file_name = file_name
            self.taesd = TAESDDecoder(file_name, max_batch_size)
        self.taesd.max_batch_size = max_batch_size

        return (self.taesd, )