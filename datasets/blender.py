import torch
import torchvision.transforms
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .img_color_augmentation import color_transformation_multi,color_transformation_single

from .ray_utils import *

import cv2

DEBUG=True







class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800),chromatic=True):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()
        self.chromatic=chromatic

        self.read_meta()
        self.white_back = True




    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []

            self.all_rgbs_2=[]
            self.chroma_codes=[]
            if self.chromatic:
                image_set_size=len(self.meta['frames'])
                self.Hue_offsets=np.random.normal( 0,0,image_set_size) # 0.1
                #self.Hue_offsets=np.random.uniform(-0.5,0.5,image_set_size)

                self.Satuation_offsets=np.random.normal(0,0,image_set_size) # 0.01
                self.Value_offsets=np.random.normal(0,0,image_set_size) # 0.01


            for i,frame in enumerate(self.meta['frames']):
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img_raw = Image.open(image_path) # PIL object
                img = img_raw.resize(self.img_wh, Image.LANCZOS) # PIL object





                img_tensor=self.transform(img)

                img_tensor_RGB = img_tensor[:3,:,:]*img_tensor[-1:,:, :]  + (1-img_tensor[-1:,:,:]) #  3 , h, w Tensor

                trans=torchvision.transforms.ToPILImage()
                img_RGB =trans(img_tensor_RGB)
                #img_RGB.show()

                img_view = img_tensor_RGB.view(3, -1).permute(1, 0) # (h*w, 4) RGBA
                #[https://blog.csdn.net/qq_19707521/article/details/78367617](https://blog.csdn.net/qq_19707521/article/details/78367617)




                '''
                
                # change the dimension
                img_tensor_rgb = img.permute((1,2,0)) # h, w ,4 Tensor
                #img_tensor_shape=img_tensor.shape
                #blend A to RGB

                #img_flatten=img.view(4,-1).permute(1,0)
                img_tensor_ = img[:3,:,:]*img[-1:,:, :]  + (1-img[-1:,:,:]) #  3 , h, w Tensor

                img_view=img_tensor_.view(3,-1).permute(1, 0) # (h*w, 3) RGBA


                img_numpy=img_tensor_rgb.permute().numpy().astype('uint8')
                '''

                dh=self.Hue_offsets[i]
                ds=self.Satuation_offsets[i]
                dv=self.Value_offsets[i]


                imgRGB_nparray=np.asarray(img_RGB)

                chromatic_img_array=color_transformation_single(imgRGB_nparray,
                                                          Hue_offset=dh,
                                                          gamma_S=ds,
                                                          gamma_V=dv)
                chromatic_img=Image.fromarray(chromatic_img_array)
                #chromatic_img.show()
                chromatic_img=self.transform(chromatic_img) # 3, H, W tensor


                chromatic_img_flatten=chromatic_img.view(3,-1).permute(1,0) # (h*w, 3), ok
                self.all_rgbs += [img_view]
                self.all_rgbs_2 +=[chromatic_img_flatten]



                self.chroma_codes_tiny =torch.tensor([dh,ds,dv])
                self.chroma_codes_tiny=self.chroma_codes_tiny.unsqueeze(0)


                self.chroma_codes +=[self.chroma_codes_tiny.repeat(img_view.shape[0],1)]


                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs_2 = torch.cat(self.all_rgbs_2, 0) # (len(self.meta['frames])*h*w, 3)
            self.chroma_codes = torch.cat(self.chroma_codes, 0) # (len(self.meta['frames])*h*w, 3)

            assert self.all_rays.shape[0]==self.all_rgbs.shape[0]
            assert self.all_rgbs.shape[0]==self.all_rgbs_2.shape[0]
            assert self.chroma_codes.shape[0]==self.all_rgbs.shape[0]
            print(f'blender dataset initialized')
    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'rgbs_2': self.all_rgbs_2[idx],
                      'chroma':self.chroma_codes[idx]
                      }

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample