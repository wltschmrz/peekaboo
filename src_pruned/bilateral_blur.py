import torch
import itertools
import rp

__all__=['BilateralProxyBlur']

def nans_like(tensor: torch.Tensor) -> torch.Tensor:
   return torch.full_like(tensor, float('nan'))

def shifted_image(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
   _, h, w = image.shape
   if abs(dx) >= w or abs(dy) >= h:
       return nans_like(image)

   # Shift x
   if dx != 0:
       nans = nans_like(image[:, :, :abs(dx)])
       output = torch.cat((nans, image[:, :, :-dx]), dim=2) if dx > 0 else \
               torch.cat((image[:, :, -dx:], nans), dim=2)
   else:
       output = image

   # Shift y 
   if dy != 0:
       nans = nans_like(image[:, :abs(dy), :])
       output = torch.cat((nans, output[:, :-dy, :]), dim=1) if dy > 0 else \
               torch.cat((output[:, -dy:, :], nans), dim=1)

   return output

def get_weight_matrix(image: torch.Tensor, sigma: float, kernel_size: int, tolerance: float):
   C, H, W = image.shape
   R = kernel_size // 2
   device, dtype = image.device, image.dtype
   
   # Get gaussian kernel
   kernel = torch.tensor(rp.gaussian_kernel(kernel_size, sigma, dim=2), 
                        dtype=dtype, device=device)
   
   # Calculate shifts for each pixel
   shifts = torch.empty(kernel_size, kernel_size, C, H, W, dtype=dtype, device=device)
   for u, v in itertools.product(range(kernel_size), range(kernel_size)):
       shifts[u, v] = shifted_image(image, u-R, v-R)
           
   # Calculate color distances and weights
   color_deltas = shifts - image[None, None]
   color_dists = (color_deltas**2).sum(dim=2).sqrt()
   color_weights = torch.exp(-0.5 * (color_dists/tolerance)**2)
   
   # Combine spatial and color weights
   weights = kernel[:,:,None,None] * color_weights
   weights = weights.nan_to_num()
   weights = weights / weights.sum((0,1), keepdim=True)
   
   return weights

def apply_weight_matrix(image: torch.Tensor, weights: torch.Tensor, iterations: int = 1):
   if iterations > 1:
       for _ in range(iterations):
           image = apply_weight_matrix(image, weights)
       return image
   
   device, dtype = image.device, image.dtype
   C, H, W = image.shape
   K = weights.shape[0]
   R = K // 2
   
   weighted_colors = torch.empty(K, K, C, H, W, dtype=dtype, device=device)
   
   for u, v in itertools.product(range(K), range(K)):
       shift = shifted_image(image, u-R, v-R).nan_to_num()
       weighted_colors[u, v] = shift * weights[u, v][None,:,:]
           
   return weighted_colors.sum((0, 1))
    
class BilateralProxyBlur:
   def __init__(self, 
                image: torch.Tensor, 
                kernel_size: int = 5,
                tolerance: float = .08,
                sigma: float = 5,
                iterations: int = 10):
       self.weights = get_weight_matrix(image, sigma, kernel_size, tolerance)
       self.iterations = iterations
       
   def __call__(self, image: torch.Tensor) -> torch.Tensor:
       return apply_weight_matrix(image, self.weights, self.iterations)