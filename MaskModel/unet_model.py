from .unet_parts import *
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, tar_den = 0.1, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.bilinear     = bilinear
        self.tar_den      = tar_den

        self.inc = (DoubleConv(in_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        # self.down4 = (Down(512, 1024 // factor))
        # self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, out_channels))


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        prob = torch.special.expit(logits)
        prob = self.scaleDensity(prob)
        return prob

    def scaleDensity(self, inp):
        b, c, h, w = inp.shape
        hw = h*w
        tar_den = self.tar_den
        curr_den = torch.norm(inp, p = 1, dim = (2, 3)).view(b, c, 1, 1) / (hw)
        return torch.where(curr_den > tar_den, inp / (curr_den + 1e-8) * tar_den, inp)


