import torch
import torch.nn as nn
import numpy as np
import einops
import rp

class GaussianFourierFeatureTransform(nn.Module):
    """Gaussian Fourier feature mapping: https://arxiv.org/abs/2006.10739
    Higher scale = higher frequency features = higher fidelity but potential noise
    Input: [B, C, H, W] -> Output: [B, 2*num_features, H, W]"""

    def __init__(self, num_channels: int, num_features: int = 256, scale: float = 10):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.freqs = nn.Parameter(torch.randn(num_channels, num_features) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        # [B, C, H, W] -> [(B*H*W), C] -> [(B*H*W), F] -> [B, H, W, F] -> [B, F, H, W]
        x = x.permute(0, 2, 3, 1).reshape(-1, self.num_channels)
        x = x @ self.freqs
        x = x.view(batch_size, height, width, self.num_features).permute(0, 3, 1, 2)
        
        # Apply sinusoidal activation
        x = 2 * torch.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)      
    

def get_uv_grid(height: int, width: int, batch_size: int = 1) -> torch.Tensor:
   """Creates UV coordinate grid of shape [B, 2, H, W] ranging [0, 1)"""
   
   coords = [np.linspace(0, 1, size, endpoint=False) for size in [height, width]]
   
   uv_grid = np.stack(np.meshgrid(*coords), -1)
   uv_grid = (torch.tensor(uv_grid)
              .unsqueeze(0)
              .permute(0, 3, 1, 2)
              .float()
              .contiguous()
              .repeat(batch_size, 1, 1, 1))
   
   return uv_grid

######## LEARNABLE IMAGES ########

class LearnableImage(nn.Module):
    def __init__(self, height: int, width: int, num_channels: int):
        super().__init__()
        self.height = height
        self.width = width 
        self.num_channels = num_channels
    
    def as_numpy_image(self) -> np.ndarray:
        return rp.as_numpy_array(self()).transpose(1, 2, 0)

    
class NoParamsDecoderWrapper(nn.Module):
   def __init__(self, decoder: nn.Module):
       super().__init__()
       self.decoder = decoder
       
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.decoder(x)
       
   def parameters(self):
       return iter(())


class LearnableLatentImage(nn.Module):
   def __init__(self, 
                learnable_image: LearnableImage,
                decoder: nn.Module,
                freeze_decoder: bool = True):
       super().__init__()
       self.learnable_image = learnable_image
       self.decoder = NoParamsDecoderWrapper(decoder) if freeze_decoder else decoder

   def forward(self) -> torch.Tensor:
       return self.decoder(self.learnable_image())


class LearnableImageRasterSigmoided(LearnableImage):
   def __init__(self, height: int, width: int, num_channels: int = 3):
       super().__init__(height, width, num_channels)
       self.image = nn.Parameter(torch.randn(num_channels, height, width))
       
   def forward(self) -> torch.Tensor:
       return torch.sigmoid(self.image)
    

class LearnableImageRaster(LearnableImage):
   def __init__(self, height: int, width: int, num_channels: int = 3):
       super().__init__(height, width, num_channels)
       self.image = nn.Parameter(torch.randn(num_channels, height, width))
       
   def forward(self) -> torch.Tensor:
       return self.image
   

class LearnableImageMLP(LearnableImage):
   def __init__(self, height: int, width: int, num_channels: int = 3, hidden_dim: int = 256):
       super().__init__(height, width, num_channels)
       
       self.uv_grid = nn.Parameter(get_uv_grid(height, width, batch_size=1), requires_grad=False)
       
       self.model = nn.Sequential(
           nn.Conv2d(2, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
           nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
           nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
           nn.Conv2d(hidden_dim, num_channels, 1),
           nn.Sigmoid(),
       )
           
   def forward(self) -> torch.Tensor:
       return self.model(self.uv_grid).squeeze(0)
    
    
class LearnableImageFourier(LearnableImage):
   def __init__(self,
                height: int = 256,
                width: int = 256, 
                num_channels: int = 3,
                hidden_dim: int = 256,
                num_features: int = 128,
                scale: int = 10):
       super().__init__(height, width, num_channels)
       
       self.num_features = num_features
       self.uv_grid = nn.Parameter(get_uv_grid(height, width, batch_size=1), requires_grad=False)
       
       self.feature_extractor = GaussianFourierFeatureTransform(2, num_features, scale)
       self.features = nn.Parameter(self.feature_extractor(self.uv_grid), requires_grad=False)
       
       feature_dim = 2 * num_features
       self.model = nn.Sequential(
           nn.Conv2d(feature_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
           nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
           nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
           nn.Conv2d(hidden_dim, num_channels, 1),
           nn.Sigmoid(),
       )
   
   def get_features(self, condition=None):
       if condition is None:
           return self.features
           
       features = einops.rearrange(self.features.clone(), 'b c h w -> b h w c')
       features[..., :len(condition)] = condition
       return einops.rearrange(features, 'b h w c -> b c h w')
       
   def forward(self, condition=None):
       return self.model(self.get_features(condition)).squeeze(0)
    
######## TEXTURE PACKS ########

class LearnableTexturePack(nn.Module):
   def __init__(self, 
                height: int, 
                width: int, 
                num_channels: int,
                num_textures: int,
                get_learnable_image):
       super().__init__()
       self.height = height
       self.width = width
       self.num_channels = num_channels
       self.num_textures = num_textures
       self.learnable_images = nn.ModuleList([
           get_learnable_image() for _ in range(num_textures)
       ])
   
   def as_numpy_images(self):
       return [x.as_numpy_image() for x in self.learnable_images]
      
   def forward(self):
       return torch.stack([x() for x in self.learnable_images])

   def __len__(self):
       return len(self.learnable_images)


class LearnableTexturePackRaster(LearnableTexturePack):
   def __init__(self, height: int = 256, width: int = 256, 
                num_channels: int = 3, num_textures: int = 1):
       super().__init__(
           height, width, num_channels, num_textures,
           lambda: LearnableImageRaster(height, width, num_channels)
       )
       
       
class LearnableTexturePackMLP(LearnableTexturePack):
   def __init__(self, height: int = 256, width: int = 256,
                num_channels: int = 3, hidden_dim: int = 256, 
                num_textures: int = 1):
       super().__init__(
           height, width, num_channels, num_textures,
           lambda: LearnableImageMLP(height, width, num_channels, hidden_dim)
       )
       self.hidden_dim = hidden_dim
       
       
class LearnableTexturePackFourier(LearnableTexturePack):
   def __init__(self, height: int = 256, width: int = 256,
                num_channels: int = 3, hidden_dim: int = 256,
                num_features: int = 128, scale: int = 10,
                num_textures: int = 1):
       super().__init__(
           height, width, num_channels, num_textures,
           lambda: LearnableImageFourier(height, width, num_channels, 
                                       hidden_dim, num_features, scale)
       )
       self.hidden_dim = hidden_dim
       self.num_features = num_features
       self.scale = scale    


class LearnableImageRasterBilateral(LearnableImageRaster):
   def __init__(self, bilateral_blur, num_channels: int = 3):
       _, height, width = bilateral_blur.image.shape
       super().__init__(height, width, num_channels)
       self.bilateral_blur = bilateral_blur
   
   def forward(self):
       return torch.sigmoid(self.bilateral_blur(self.image))
   
   
class LearnableImageFourierBilateral(LearnableImageFourier):
   def __init__(self, bilateral_blur, num_channels: int = 3,                  
                hidden_dim: int = 256,
                num_features: int = 128,
                scale: int = 10):
       _, height, width = bilateral_blur.image.shape
       super().__init__(height, width, num_channels, hidden_dim, num_features, scale)
       
       feature_dim = 2 * num_features
       self.model = nn.Sequential(
           nn.Conv2d(feature_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
           nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim), 
           nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
           nn.Conv2d(hidden_dim, num_channels, 1)
       )
       self.bilateral_blur = bilateral_blur

   def forward(self, condition=None):
       return torch.sigmoid(self.bilateral_blur(
           self.model(self.get_features(condition)).squeeze(0)
       ))
    
    
class LearnableAlphasFourier(LearnableImage):
   def __init__(self,
                height: int = 256,
                width: int = 256,
                num_channels: int = 3,
                hidden_dim: int = 256,
                num_features: int = 128,
                scale: int = 10):
       super().__init__(height, width, num_channels)
       
       self.num_features = num_features
       self.uv_grid = nn.Parameter(get_uv_grid(height, width, batch_size=1), requires_grad=False)
       self.feature_extractor = GaussianFourierFeatureTransform(2, num_features, scale)
       self.features = nn.Parameter(self.feature_extractor(self.uv_grid), requires_grad=False)
       
       feature_dim = 2 * num_features
       self.model = nn.Sequential(
           nn.Conv2d(feature_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
           nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
           nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(), nn.BatchNorm2d(hidden_dim),
       )
       
       self.gate_net = nn.Conv2d(hidden_dim, 1, 1)
       self.selector_net = nn.Conv2d(hidden_dim, num_channels, 1)
   
   def get_features(self, condition=None):
       if condition is None:
           return self.features
           
       features = einops.rearrange(self.features.clone(), 'b c h w -> b h w c')
       features[..., :len(condition)] = condition
       return einops.rearrange(features, 'b h w c -> b c h w')

   def forward(self, condition=None):
       output = self.model(self.get_features(condition))
       
       gate = torch.sigmoid(self.gate_net(output))
       select = torch.sigmoid(self.selector_net(output))
       select = select / select.sum(dim=1, keepdim=True)
       
       return (gate * select).squeeze(0)