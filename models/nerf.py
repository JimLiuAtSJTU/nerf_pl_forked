import copy
import time
import warnings

import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        #print('x shape \n')
        #print(x.shape)
        if len(x.shape)!=2:
            time.sleep(2)
            raise AssertionError

        assert x.shape[1]==3
        # the original implementation might have some problems
        out = torch.zeros((x.shape[0],self.out_channels))
        #print(f'out{out.shape}')
        out[:,0:self.in_channels]=x

        start=self.in_channels


        for i in range(len(self.freq_bands)):

            freq=self.freq_bands[i]
            #print(f'freq{freq}')

            for j,func in enumerate(self.funcs):
                mapped_=func(x*freq)
                #print(mapped_.shape)
                out[:,start:start+self.in_channels]=mapped_
                start= start + self.in_channels

        assert start==self.out_channels
        return out

        
        '''

        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)
        '''

class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out




class NeRF_Chromatic(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27,
                 in_channels_chroma_code=27,
                 skips=[4],
                 default_chroma_codes=None):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_chroma_code : number of input channels for color transformation code (3+3*4*2=27 by default)

        skips: add skip connection in the Dth layer
        """
        super(NeRF_Chromatic, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_chromatrans = in_channels_chroma_code

        if default_chroma_codes is None:
            warnings.warn('caution. use default chroma code, i.e. embedded{zeros(3)}')
            embedding_func=Embedding(3,4)
            self.default_chroma_code=embedding_func(torch.zeros(1,3))
        else:
            self.default_chroma_code=default_chroma_codes

        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W + in_channels_dir
                                          + in_channels_chroma_code
                                          , W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)

        self.rgb_residial_block=nn.Sequential(
            nn.Linear(W // 2, W //2 ),
            nn.ReLU()
        )

        self.rgb_residial_block2=copy.deepcopy(self.rgb_residial_block)


        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """


        '''
        for training step, there are chroma code.
        
        ### IMPORTANT ###
        
        the output dimension of train and inference is different. 
        see the forward_inference function for details.
        
        '''
        if x.shape[-1]>(self.in_channels_xyz+self.in_channels_dir):
            # training.
            return self.forward_train(x,sigma_only)
        else:
            # inference
            return self.forward_inference(x,sigma_only)
    def forward_train(self,x,sigma_only):

        if not sigma_only:
            input_xyz, input_dir,input_chroma_code1,input_chroma_code2 = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir,
                                self.in_channels_chromatrans,
                                self.in_channels_chromatrans
                                ], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)# xyz -> sigma
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        # color network , called dir_encoding function here.
        # [xyz_encoding,   direction, color_chroma ] ->  rgb
        #



        rgb1 = self.color_network(xyz_encoding_final, input_dir,input_chroma_code1)

        rgb2 = self.color_network(xyz_encoding_final, input_dir,input_chroma_code2)


        out = torch.cat([rgb1, rgb2, sigma], -1)

        return out


    def forward_inference(self,x,sigma_only):


        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir,
                                ], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        sigma = self.sigma(xyz_)  # xyz -> sigma
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        # color network , called dir_encoding function here.
        # [xyz_encoding,   direction, color_chroma ] ->  rgb
        #

        rgb1 = self.color_network(xyz_encoding_final, input_dir, self.default_chroma_code)


        '''
        IMPORTANT!!! 
        for the output in inference stage, simply use the output just like the original NeRF.
        
        to avoid modifying the pipelines, dimension settings in the code related to the inference part. 
        
        '''
        out = torch.cat([rgb1, sigma], -1)

        return out

    def color_network(self,xyz_encoding_final,input_dir,input_chroma_code):

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir, input_chroma_code], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        dir_encoding_2 = self.rgb_residial_block(dir_encoding)
        #rgb = self.rgb_residial_block2(dir_encoding+dir_encoding_2)


        #use residual connection seems to lead to not converge problem
        rgb = self.rgb_residial_block2(dir_encoding_2)

        rgb = self.rgb(rgb)
        return rgb


