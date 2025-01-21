from typing import Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import (
   AutoencoderKL,
   PNDMScheduler,
   StableDiffusionPipeline,
   UNet2DConditionModel,
)
from transformers import (
   CLIPTextModel,
   CLIPTokenizer,
   logging,
)

# Suppress partial model loading warning
logging.set_verbosity_error()

class StableDiffusion(nn.Module):
    VAE_SCALING_FACTOR = 0.18215

    def __init__(self, device='cuda', checkpoint_path="CompVis/stable-diffusion-v1-4"):
        super().__init__()
        self.device = torch.device(device)
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        
        print('[INFO] sd.py: loading stable diffusion...please make sure you have run `huggingface-cli login`.')
        
        # Initialize pipeline with PNDM scheduler (load from pipeline. let us use dreambooth models.)
        pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint_path, 
            torch_dtype=torch.float,
            scheduler=PNDMScheduler(  # Error from scheduling_lms_discrete.py
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=self.num_train_timesteps
            )
        )
        pipe.safety_checker = lambda images, _: (images, False)  # Disable the NSFW checker (slows things down)
        
        # Setup components and move to device
        self.pipe = pipe
        self.components = {
            'vae': (pipe.vae, AutoencoderKL),
            'tokenizer': (pipe.tokenizer, CLIPTokenizer),
            'text_encoder': (pipe.text_encoder, CLIPTextModel),
            'unet': (pipe.unet, UNet2DConditionModel),
            'scheduler': (pipe.scheduler, PNDMScheduler)
        }
        
        # Initialize and validate components
        for name, (component, expected_type) in self.components.items():
            if name in ['vae', 'text_encoder', 'unet']:
                component = component.to(self.device)
            assert isinstance(component, expected_type), f"{name} type mismatch: {type(component)}"
            setattr(self, name, component)
        
        self.uncond_text = ''
        self.checkpoint_path = checkpoint_path
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] sd.py: loaded stable diffusion!')

    def get_text_embeddings(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        prompts = [prompts] if isinstance(prompts, str) else prompts

        def get_embeddings(text_list):
            tokens = self.tokenizer(
                text_list,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids
            
            with torch.no_grad():
                return self.text_encoder(tokens.to(self.device))[0]

        # Get text and unconditional embeddings
        text_embeddings = get_embeddings(prompts)  # [B, 77, 768]
        uncond_embeddings = get_embeddings([self.uncond_text] * len(prompts))  # [B, 77, 768]

        assert (uncond_embeddings == uncond_embeddings[0][None]).all()  # All the same
        
        return torch.cat([uncond_embeddings, text_embeddings])  # First B rows: uncond, Last B rows: cond

    def train_step(
    self, text_embeddings: torch.Tensor, pred_rgb: torch.Tensor, 
    guidance_scale: float = 100, t: Optional[int] = None
    ):
        """Generate dream-loss gradients. Main training step for image generation."""        
        
        # Prepare image and timestep
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        t = t if t is not None else torch.randint(self.min_step, self.max_step + 1, [1], device=self.device)
        assert 0 <= t < self.num_train_timesteps, f'invalid timestep t={t}'

        # Encode image to latents (with grad)
        latents = self.encode_imgs(pred_rgb_512)

        # # Predict noise without grad
        # with torch.no_grad():
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t.cpu())
        noise_pred = self.unet(torch.cat([latents_noisy] * 2), t, encoder_hidden_states=text_embeddings)['sample']

        # Guidance . High value from paper
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Calculate and apply gradients
        w = (1 - self.alphas[t])
        loss = w * ((noise_pred - noise) ** 2).mean()
        loss.backward(retain_graph=True)
        # grad = w * (noise_pred - noise)
        # latents.backward(gradient=grad, retain_graph=True)
        return loss.item()  # dummy loss value

    def encode_imgs(self, imgs:torch.Tensor)->torch.Tensor:
        imgs = 2 * imgs - 1  # [-1, 1]
        posterior = self.vae.encode(imgs)
        latents = posterior.latent_dist.sample() * self.VAE_SCALING_FACTOR  # [B, 3, H, W]
        return latents
