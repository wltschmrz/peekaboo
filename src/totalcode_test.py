'''
현 test 요소: 
    alpha를 mean으로 조정
    SDS는 abs로 부여 및 Autograd
'''

import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src')
sys.path.extend([proj_dir, src_dir])

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import PNDMScheduler, StableDiffusionPipeline
from transformers import logging
import rp
from src.bilateral_blur import BilateralProxyBlur
from src.learnable_textures import (
    LearnableImageFourier,
    LearnableImageFourierBilateral,
    LearnableImageRaster,
    LearnableImageRasterBilateral,
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
            scheduler=PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        )
        pipe.safety_checker = lambda images, _: (images, False)  # Disable the NSFW checker (slows things down)
        
        self.pipe = pipe
        self.components = {
            'vae': pipe.vae,
            'tokenizer': pipe.tokenizer,
            'text_encoder': pipe.text_encoder,
            'unet': pipe.unet,
            'scheduler': pipe.scheduler,
        }
        
        # Initialize and validate components
        for name, component in self.components.items():
            if name in ['vae', 'text_encoder', 'unet']:
                component = component.to(self.device)
            setattr(self, name, component)
        
        self.uncond_text = ''
        self.checkpoint_path = checkpoint_path
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        print(f'[INFO] sd.py: loaded stable diffusion!')

    def get_text_embeddings(self, prompts) -> torch.Tensor:
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

    def train_step(self, text_embeddings: torch.Tensor, pred_rgb: torch.Tensor, guidance_scale: float = 100, t = None):
        """Generate dream-loss gradients. Main training step for image generation."""        
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        t = t if t is not None else torch.randint(self.min_step, self.max_step + 1, [1], device=self.device)
        assert 0 <= t < self.num_train_timesteps, f'invalid timestep t={t}'

        latents = self.encode_imgs(pred_rgb_512)

        # with torch.no_grad():
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t.cpu())
        noise_pred_cat = self.unet(torch.cat([latents_noisy] * 2), t, encoder_hidden_states=text_embeddings)['sample']
        noise_pred_uncond, noise_pred_text = noise_pred_cat.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w = (1 - self.alphas[t])
        # loss = w * ((noise_pred - noise) ** 2).mean()
        loss = torch.abs(noise - noise_pred).sum()
        # loss.backward(retain_graph=True)
        # grad = w * (noise_pred - noise)
        # latents.backward(gradient=grad, retain_graph=True)
        return loss

    def encode_imgs(self, imgs:torch.Tensor)->torch.Tensor:
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs)
        latents = posterior.sample() * self.VAE_SCALING_FACTOR
        return latents

sd = StableDiffusion('cuda','CompVis/stable-diffusion-v1-4')
device = sd.device

def make_learnable_image(height, width, num_channels, foreground=None, bilateral_kwargs={}, representation='fourier'):
    bilateral_blur = BilateralProxyBlur(foreground, **bilateral_kwargs)
    image_types = {
        'fourier bilateral': LearnableImageFourierBilateral(bilateral_blur, num_channels),
        'raster bilateral': LearnableImageRasterBilateral(bilateral_blur, num_channels),
        'fourier': LearnableImageFourier(height, width, num_channels),
        'raster': LearnableImageRaster(height, width, num_channels)
    }
    return image_types.get(representation, ValueError(f'Invalid method: {representation}'))

def blend_torch_images(foreground, background, alpha):
    return foreground * alpha + background * (1 - alpha)

def make_image_square(image: np.ndarray, method='crop') -> np.ndarray:
    image = rp.as_rgb_image(image)
    min_dim = min(image.shape[:2])
    if method == 'crop':
        image = rp.crop_image(image, min_dim, min_dim, origin='center')
    return rp.resize_image(image, (512, 512))

class PeekabooSegmenter(nn.Module):
    def __init__(self,
                 image: np.ndarray,
                 labels,  # List[BaseLabel]
                 size: int = 256,
                 name: str = 'Untitled',
                 bilateral_kwargs: dict = None,
                 representation: str = 'fourier bilateral',
                 min_step = None,
                 max_step = None):
        super().__init__()
        
        self.height = self.width = size  # use square img for now
        self.labels = labels
        self.name = name
        self.representation = representation
        self.min_step = min_step
        self.max_step = max_step
        
        self.image = self._preprocess_image(image)        
        self.foreground = rp.as_torch_image(image).to(device)  # Convert image to tensor in CHW form        
        self.background = torch.zeros_like(self.foreground)  # background is solid color now
        self.alphas = make_learnable_image(
            self.height, self.width,
            num_channels=len(self.labels),
            foreground=self.foreground,
            representation=representation,
            bilateral_kwargs=bilateral_kwargs or {}
        )            

    def _preprocess_image(self, image):
        image = rp.cv_resize_image(image, (self.height, self.width))
        image = rp.as_rgb_image(image)  # 3 channels in HWC form
        return rp.as_float_image(image)  # value in [0, 1]

    def set_background_color(self, color):
        self.background[:] = torch.tensor(color, device=device).view(3, 1, 1)
        
    def randomize_background(self):
        self.set_background_color(rp.random_rgb_float_color())
        
    def forward(self, alphas=None, return_alphas=False):
        old_min_step, old_max_step = sd.min_step, sd.max_step
        try:
            sd.min_step, sd.max_step = self.min_step, self.max_step
            alphas = alphas if alphas is not None else self.alphas()
            output_images = torch.stack([blend_torch_images(self.foreground, self.background, alpha) for alpha in alphas])
            return (output_images, alphas) if return_alphas else output_images
        finally:
            sd.min_step, sd.max_step = old_min_step, old_max_step

class SimpleLabel:
    def __init__(self, name:str):
        self.name=name  # :str
        self.embedding=sd.get_text_embeddings(name).to(device)  # :torch.Tensor                

def run_peekaboo(name: str,
                 image,  # Union[str, np.ndarray]
                 label=None,  # Optional[BaseLabel]
                 GRAVITY=1e-1/2,      # prompt에 따라 tuning이 제일 필요. (1e-2, 1e-1/2, 1e-1, 1.5*1e-1)
                 NUM_ITER=300,        # 이정도면 충분
                 LEARNING_RATE=1e-5,  # neural neural texture 아니면 키워도 됨.
                 BATCH_SIZE=1,        # 키우면 vram만 잡아먹음
                 GUIDANCE_SCALE=100,  # DreamFusion 참고하여 default값 설정
                 bilateral_kwargs=None,
                 square_image_method='crop',
                 representation='fourier bilateral',
                 min_step=None,
                 max_step=None):
    
    bilateral_kwargs = bilateral_kwargs or {
        'kernel_size': 3,
        'tolerance': .08,
        'sigma': 5,
        'iterations': 40
    }

    label = label or SimpleLabel(name)
    image = rp.load_image(image) if isinstance(image, str) else image
    image = rp.as_rgb_image(rp.as_float_image(make_image_square(image, square_image_method)))

    p = PeekabooSegmenter(
        image=image,
        labels=[label],
        name=name,
        bilateral_kwargs=bilateral_kwargs,
        representation=representation,
        min_step=min_step,
        max_step=max_step
    ).to(device)

    optim = torch.optim.SGD(list(p.parameters()), lr=LEARNING_RATE)
    preview_interval = max(1, NUM_ITER // 10)

    def train_step():
        optim.zero_grad()
        alphas = p.alphas()
        for _ in range(BATCH_SIZE):
            p.randomize_background()
            composites = p()
            pkbooloss = 0
            for label, composite in zip(p.labels, composites):
                sdsloss = sd.train_step(label.embedding, composite[None], guidance_scale=GUIDANCE_SCALE)
                pkbooloss += sdsloss

        alphaloss = alphas.mean() * GRAVITY
        pkbooloss += alphaloss
        pkbooloss.backward()
        optim.step()
        return sdsloss.item(), alphaloss.item()

    l1, l2 = [], []
    try:
        for i in range(NUM_ITER):
            sdsloss, alphaloss = train_step()
            l1.append(sdsloss)
            l2.append(alphaloss)
            if not i % preview_interval:
                with torch.no_grad():
                    pass  # preview result saving
    except KeyboardInterrupt:
        pass
    
    output_folder = rp.make_folder(f'peekaboo_results/{name}')
    output_folder_path += f'{len(rp.get_subfolders(output_folder)):03d}'
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Loss plot 저장
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(l1, label='SDS Loss')
    plt.xlabel('Iteration'); plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(l2, label='Alpha Loss')
    plt.xlabel('Iteration'); plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_folder_path}/loss_plot.png')
    plt.close()
    
    # 최종 결과 저장
    with torch.no_grad():
        final_result = p()
        for label, result in zip(p.labels, final_result):
            save_path = f'{output_folder_path}/{label.name}_result.png'
            rp.save_image(result.cpu().numpy(), save_path)
        
        # alpha mask도 저장
        alpha_mask = p.alphas().cpu().numpy()
        rp.save_image(alpha_mask, f'{output_folder_path}/alpha_mask.png')
    
    return final_result, (l1, l2)
    

if __name__ == "__main__":

    prms = {
        'G': 1e-1/2,
        'iter': 300,
        'lr': 1e-5,
        'B': 1,
        'guidance': 100,
        'representation': 'fourier bilateral',
    }

    run_peekaboo(
        name='Mario',
        image="https://i1.sndcdn.com/artworks-000160550668-iwxjgo-t500x500.jpg",
        GRAVITY=prms['G'],
        NUM_ITER=prms['iter'],
        LEARNING_RATE=prms['lr'],
        BATCH_SIZE=prms['B'],
        GUIDANCE_SCALE=prms['guidance'],
        representation=prms['representation'],
        )