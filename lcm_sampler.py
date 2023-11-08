from comfy.samplers import *
from comfy.k_diffusion.sampling import generic_step_sampler

def ksampler_lcm(sampler_name, eta, extra_options={}, inpaint_options={}):
    class KSAMPLER(Sampler):
        def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            extra_args["denoise_mask"] = denoise_mask
            model_k = KSamplerX0Inpaint(model_wrap)
            model_k.latent_image = latent_image
            if inpaint_options.get("random", False): #TODO: Should this be the default?
                generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
                model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
            else:
                model_k.noise = noise

            if self.max_denoise(model_wrap, sigmas):
                noise = noise * torch.sqrt(1.0 + sigmas[0] ** 2.0)
            else:
                noise = noise * sigmas[0]

            k_callback = None
            total_steps = len(sigmas) - 1
            if callback is not None:
                k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]

            samples = sample_lcm(model_k, noise, sigmas, eta, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **extra_options)
            return samples
    return KSAMPLER

def LCMSampler_outer(eta):
    def LCMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
        alpha_cumprod = 1 / ((sigma * sigma) + 1)
        alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)

        mu = (x - (1 - alpha_cumprod).sqrt() * noise) / alpha_cumprod.sqrt()
        if sigma_prev > 0:
            # this is not mathematically correct, but eta=1 means lcm_sampler and eta=0 means ddim(euler).
            noise_interp = (1 - eta) * noise + eta * noise_sampler(sigma, sigma_prev) 
            mu = alpha_cumprod_prev.sqrt() * mu + (1 - alpha_cumprod_prev).sqrt() * noise_interp
        return mu
    return LCMSampler_step

@torch.no_grad()
def sample_lcm(model, x, sigmas, eta, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, LCMSampler_outer(eta))

class SamplerLCM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {"eta": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, #Minimum value
                    "max": 1.0, #Maximum value
                    "step": 0.01 #Slider's step
                }),
            }
        }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta: float):
        sampler = ksampler_lcm("", eta)()
        return (sampler, )
