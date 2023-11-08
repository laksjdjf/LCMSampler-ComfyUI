from .lcm_sampler import SamplerLCM
from .taesd_decoder import TAESDLoader

NODE_CLASS_MAPPINGS = {
    "SamplerLCM": SamplerLCM,
    "TAESDLoader": TAESDLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerLCM": "Load SamplerLCM",
    "TAESDLoader": "Load TAESD",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]