from .unet_parts import *
import torch
import torch.nn as nn
from torch.nn.functional import relu

class MaskNet(nn.Module):
    def __init__(self, inp_channels, out_channels, tar_den = 0.1):
        super(MaskNet, self).__init__()
        self.tar_den = tar_den

        # Encoder
        self.e11 = nn.Conv2d(inp_channels, 32, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False) # output: 570x570x64
        self.be11 = nn.BatchNorm2d(32)
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False) # output: 568x568x64
        self.be12 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        self.e21 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False) # output: 282x282x128
        self.be21 = nn.BatchNorm2d(64)
        self.e22 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False) # output: 280x280x128
        self.be22 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        self.e31 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False) # output: 138x138x256
        self.be31 = nn.BatchNorm2d(128)
        self.e32 = nn.Conv2d(128, 128, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False) # output: 136x136x256
        self.be32 = nn.BatchNorm2d(128)
        self.e33 = nn.Conv2d(128, 128, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False) # output: 136x136x256
        self.be33 = nn.BatchNorm2d(128)



        # decoder
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False)
        self.bd31 = nn.BatchNorm2d(64)
        self.d32 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False)
        self.bd32 = nn.BatchNorm2d(64)
        

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False)
        self.bd41 = nn.BatchNorm2d(32)
        self.d42 = nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode = 'reflect', bias=False)
        self.bd42 = nn.BatchNorm2d(32)

        # Output layer
        self.outconv = nn.Conv2d(32, out_channels, kernel_size=1)
        


    def forward(self, x):
        # Encoder
        xe11 = relu(self.be11(self.e11(x)))
        xe12 = relu(self.be12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.be21(self.e21(xp1)))
        xe22 = relu(self.be22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.be31(self.e31(xp2)))
        xe32 = relu(self.be32(self.e32(xe31)))
        xe33 = relu(self.be33(self.e33(xe32)))
        # xp3 = self.pool3(xe32)

        xu3 = self.upconv3(xe33)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.bd31(self.d31(xu33)))
        xd32 = relu(self.bd32(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.bd41(self.d41(xu44)))
        xd42 = relu(self.bd42(self.d42(xd41)))

        # Output layer
        logits = self.outconv(xd42)
        prob = torch.special.expit(logits)
        prob_scaled = self.scaleDensity(prob)

        return prob_scaled
    

    def scaleDensity(self, inp):
        b, c, h, w = inp.shape
        hw = h*w
        tar_den = self.tar_den
        curr_den = torch.norm(inp, p = 1, dim = (2, 3)).view(b, c, 1, 1) / (hw)
        return torch.where(curr_den > tar_den, inp / (curr_den + 1e-8) * tar_den, inp)



