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
    def __init__(self):
        super(MSE_Enhanced_Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets_A,targets_B):
        loss = self.loss(inputs['rgb_coarse'], targets_A)
        loss += self.loss(inputs['rgb_coarse'], targets_B)

        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets_A)
            loss += self.loss(inputs['rgb_fine'], targets_B)

        return loss


loss_dict = {'mse': MSELoss,
             'enhanced':MSE_Enhanced_Loss}