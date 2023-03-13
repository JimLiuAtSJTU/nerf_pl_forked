import torch
from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss


class MSE_Enhanced_Loss(nn.Module):
    def __init__(self, lambda_=0.2):
        super(MSE_Enhanced_Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

        self.lam_ = lambda_

    def forward(self, inputs, targets_RGB_img, targets_Chroma_img):
        loss = self.loss(inputs['rgb_coarse'], targets_RGB_img)
        loss += self.loss(inputs['rgb2_coarse'], targets_Chroma_img) * self.lam_

        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets_RGB_img)
            loss += self.loss(inputs['rgb2_fine'], targets_Chroma_img) * self.lam_
        return loss/(1+self.lam_)


loss_dict = {'mse': MSELoss,
             'enhanced': MSE_Enhanced_Loss}
