import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_2')
sys.path.extend([proj_dir, src_dir])

from typing import Union, List, Optional
import numpy as np
import rp
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from easydict import EasyDict
import matplotlib.pyplot as plt

import src_2.stable_diffusion as sd
from src_2.bilateral_blur import BilateralProxyBlur
from src_2.learnable_textures import (
    LearnableImageFourier,
    LearnableImageFourierBilateral,
    LearnableImageRasterSigmoided,
    LearnableImageRasterBilateral,
    )

#Importing this module loads a stable diffusion model. Hope you have a GPU!
sd=sd.StableDiffusion('cuda','CompVis/stable-diffusion-v1-4')
device=sd.device

def make_learnable_image(height, width, num_channels, foreground=None, bilateral_kwargs={}, representation='fourier'):
    bilateral_blur = BilateralProxyBlur(foreground, **bilateral_kwargs)
    image_types = {
        'fourier bilateral': LearnableImageFourierBilateral(bilateral_blur, num_channels),
        'raster bilateral': LearnableImageRasterBilateral(bilateral_blur, num_channels),
        'fourier': LearnableImageFourier(height, width, num_channels),
        'raster': LearnableImageRasterSigmoided(height, width, num_channels)
    }
    return image_types.get(representation, ValueError(f'Invalid method: {representation}'))

def blend_torch_images(foreground, background, alpha):
    assert foreground.shape==background.shape
    C,H,W=foreground.shape
    assert alpha.shape==(H,W), 'alpha is a matrix'
    return foreground*alpha + background*(1-alpha)

class PeekabooSegmenter(nn.Module):
    def __init__(self,
                 image:np.ndarray,
                 labels:List['BaseLabel'],
                 size:int=256,
                 name:str='Untitled',
                 bilateral_kwargs:dict={},
                 representation = 'fourier bilateral',
                 min_step=None,
                 max_step=None,
                ):
        
        super().__init__()
        
        height=width=size #We use square images for now
        
        assert all(issubclass(type(label),BaseLabel) for label in labels)
        assert len(labels), 'Must have at least one class to segment'
        
        self.height=height
        self.width=width
        self.labels=labels
        self.name=name
        self.representation=representation
        self.min_step=min_step
        self.max_step=max_step
        
        assert rp.is_image(image), 'Input should be a numpy image'
        image=rp.cv_resize_image(image,(height,width))
        image=rp.as_rgb_image(image) #Make sure it has 3 channels in HWC form
        image=rp.as_float_image(image) #Make sure it's values are between 0 and 1
        assert image.shape==(height,width,3) and image.min()>=0 and image.max()<=1
        self.image=image
        self.foreground=rp.as_torch_image(image).to(device) #Convert the image to a torch tensor in CHW form
        assert self.foreground.shape==(3, height, width)
        self.background=self.foreground*0 #The background will be a solid color for now
        self.alphas=make_learnable_image(height,width,num_channels=self.num_labels,foreground=self.foreground,representation=self.representation,bilateral_kwargs=bilateral_kwargs)
            
    @property
    def num_labels(self):
        return len(self.labels)
            
    def set_background_color(self, color):
        r,g,b = color
        assert 0<=r<=1 and 0<=g<=1 and 0<=b<=1
        self.background[0]=r
        self.background[1]=g
        self.background[2]=b
        
    def randomize_background(self):
        self.set_background_color(rp.random_rgb_float_color())
        
    def forward(self, alphas=None, return_alphas=False):
        try:
            old_min_step, old_max_step = sd.min_step, sd.max_step
            if (self.min_step is not None) and (self.max_step is not None):
                sd.min_step, sd.max_step = self.min_step, self.max_step
            output_images = []
            if alphas is None:
                alphas=self.alphas()
            assert alphas.shape==(self.num_labels, self.height, self.width)
            assert alphas.min()>=0 and alphas.max()<=1
            for alpha in alphas:
                output_image=blend_torch_images(foreground=self.foreground, background=self.background, alpha=alpha)
                output_images.append(output_image)
            output_images=torch.stack(output_images)
            assert output_images.shape==(self.num_labels, 3, self.height, self.width) #In BCHW form
            if return_alphas:
                return output_images, alphas
            else:
                return output_images
        finally:
            sd.min_step = old_min_step
            sd.max_step = old_max_step

def get_mean_embedding(prompts:list):
    return torch.mean(torch.stack([sd.get_text_embeddings(prompt) for prompt in prompts]), dim=0).to(device)

class BaseLabel:
    def __init__(self, name:str, embedding:torch.Tensor):
        self.name=name
        self.embedding=embedding
        
    def get_sample_image(self):
        output=sd.embeddings_to_imgs(self.embedding)[0]
        assert rp.is_image(output)
        return output

    def __repr__(self):
        return '%s(name=%s)'%(type(self).__name__,self.name)
        
class SimpleLabel(BaseLabel):
    def __init__(self, name:str):
        super().__init__(name, sd.get_text_embeddings(name).to(device))

class PeekabooResults(EasyDict):
    #Acts like a dict, except you can read/write parameters by doing self.thing instead of self['thing']
    pass

def save_peekaboo_results(results,new_folder_path):
    assert not rp.folder_exists(new_folder_path), 'Please use a different name, not %s'%new_folder_path
    rp.make_folder(new_folder_path)
    with rp.SetCurrentDirectoryTemporarily(new_folder_path):
        print("Saving PeekabooResults to "+new_folder_path)
        params={}
        for key in results:
            value=results[key]
            if rp.is_image(value): 
                #Save a single image
                rp.save_image(value,key+'.png')
            elif isinstance(value, np.ndarray) and rp.is_image(value[0]):
                #Save a folder of images
                rp.make_directory(key)
                with rp.SetCurrentDirectoryTemporarily(key):
                    for i in range(len(value)):
                        rp.save_image(value[i],str(i)+'.png')
            elif isinstance(value, np.ndarray):
                #Save a generic numpy array
                np.save(key+'.npy',value) 
            else:

                import json
                try:
                    json.dumps({key:value})
                    #Assume value is json-parseable
                    params[key]=value
                except Exception:
                    params[key]=str(value)
        rp.save_json(params,'params.json',pretty=True)
        print("Done saving PeekabooResults to "+new_folder_path+"!")
        
def make_image_square(image:np.ndarray, method='crop')->np.ndarray:
    #Takes any image and makes it into a 512x512 square image with shape (512,512,3)
    assert rp.is_image(image)
    assert method in ['crop','scale']
    image=rp.as_rgb_image(image)
    
    height, width = rp.get_image_dimensions(image)
    min_dim=min(height,width)
    max_dim=max(height,width)
    
    if method=='crop':
        return make_image_square(rp.crop_image(image, min_dim, min_dim, origin='center'),'scale')
    if method=='scale':
        return rp.resize_image(image, (512,512))
                    

def run_peekaboo(name:str, image:Union[str,np.ndarray], label:Optional['BaseLabel']=None,
                
                #Peekaboo Hyperparameters:
                GRAVITY=1e-1/2, # This is the one that needs the most tuning, depending on the prompt...
                #   ...usually one of the following GRAVITY will work well: 1e-2, 1e-1/2, 1e-1, or 1.5*1e-1
                NUM_ITER=300,       # 300 is usually enough
                LEARNING_RATE=1e-5, # Can be larger if not using neural neural textures (aka when representation is raster)
                BATCH_SIZE=1,       # Doesn't make much difference, larger takes more vram
                GUIDANCE_SCALE=100, # The defauly value from the DreamFusion paper
                bilateral_kwargs=dict(kernel_size = 3,
                                      tolerance = .08,
                                      sigma = 5,
                                      iterations=40,
                                     ),
                square_image_method='crop', #Can be either 'crop' or 'scale' - how will we square the input image?
                representation='fourier bilateral', #Can be 'fourier bilateral', 'raster bilateral', 'fourier', or 'raster'
                min_step=None,
                max_step=None,
                )->PeekabooResults:
    
    if label is None: 
        label=SimpleLabel(name)
    
    image_path='<No image path given>'
    if isinstance(image,str):
        image_path=image
        image=rp.load_image(image)
    
    assert rp.is_image(image)
    assert issubclass(type(label),BaseLabel)
    image=rp.as_rgb_image(rp.as_float_image(make_image_square(image,square_image_method)))
    rp.tic()

    p=PeekabooSegmenter(image,
                        labels=[label],
                        name=name,
                        bilateral_kwargs=bilateral_kwargs,
                        representation=representation, 
                        min_step=min_step,
                        max_step=max_step,
                       ).to(device)

    params=list(p.parameters())
    optim=torch.optim.SGD(params,lr=LEARNING_RATE)
    # scheduler = LambdaLR(optim, lr_lambda=lambda iter_num: 1 / (1 + iter_num))

    preview_interval=NUM_ITER//10 
    preview_interval=max(1,preview_interval)

    l1, l2, noi1, noi2, noi3 = [], [], [], [], []
    try:
        display_eta=rp.eta(NUM_ITER)
        for iter_num in range(NUM_ITER):
            display_eta(iter_num)

            alphas=p.alphas()
            # for __ in range(BATCH_SIZE):
            assert BATCH_SIZE == 1, 'batchsize != 1'
            p.randomize_background()
            composites=p()

            # for label, composite in zip(p.labels, composites):
            label, composite = p.labels[0], composites[0]
            sdsloss, unc, con, cont = sd.train_step(label.embedding, composite[None], guidance_scale=GUIDANCE_SCALE)
            loss = alphas.mean() * GRAVITY
            alphaloss = loss.item()
            loss += sdsloss
            loss.backward()
            optim.step()
            optim.zero_grad()
            # scheduler.step()

            l1.append(sdsloss)
            l2.append(alphaloss)
            noi1.append(unc)
            noi2.append(con)
            noi3.append(cont)

    except KeyboardInterrupt:
        print("Interrupted early, returning current results...")
        pass

    p.set_background_color((0,0,0))

    results = PeekabooResults(
        #The main output is the alphas
        alphas=rp.as_numpy_array(alphas),
        output=rp.as_numpy_images(p(p.alphas())),
        
        #Keep track of hyperparameters used
        GRAVITY=GRAVITY,
        BATCH_SIZE=BATCH_SIZE,
        NUM_ITER=NUM_ITER,
        GUIDANCE_SCALE=GUIDANCE_SCALE,
        bilateral_kwargs=bilateral_kwargs,
        lr=LEARNING_RATE,
        representation=representation,
        
        label=label,
        image=image,
        image_path=image_path,
        
        height=p.height,
        width=p.width,
        p_name=p.name,
        
        min_step=p.min_step,
        max_step=p.max_step,
        
        device=device,
    ) 
    
    output_folder = rp.make_folder('peekaboo_results/%s'%name)
    output_folder += '/%03i'%len(rp.get_subfolders(output_folder))
    
    save_peekaboo_results(results,output_folder)
    print("Saved results at %s"%output_folder)
    
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
    plt.savefig(f'{output_folder}/loss_plot.png')
    plt.close()

    # Loss plot 저장
    plt.figure(figsize=(16, 9))
    plt.subplot(3, 1, 1)
    plt.plot(noi1, label='uncond')
    plt.xlabel('Iteration'); plt.ylabel('abs mean')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(noi2, label='cond')
    plt.xlabel('Iteration'); plt.ylabel('abs mean')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(noi3, label='cont')
    plt.xlabel('Iteration'); plt.ylabel('abs mean')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/score_plot.png')
    plt.close()

if __name__ == "__main__":
    
    # # bilateral fourier 용도.
    # prms = {
    #     'G': 3000,
    #     'iter': 300,
    #     'lr': 1e-5,
    #     'B': 1,
    #     'guidance': 100,
    #     'representation': 'fourier bilateral',
    # }

    # raster 용도.
    prms = {
        'G': 3000,
        'iter': 300,
        'lr': 1,
        'B': 1,
        'guidance': 100,
        'representation': 'raster',
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