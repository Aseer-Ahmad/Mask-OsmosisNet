import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/vasatdan/wgain-inpaint/blob/main/WGAIN_model.py

class UNet_ContextAgg(nn.Module):
    def __init__(self, inp_channels, out_channels, tar_den = 0.1,):
        super(UNet_ContextAgg, self).__init__()
        
        self.tar_den = tar_den

        # Downsampling layers (128 -> 126)
        self.conv1_1 = nn.Conv2d(inp_channels, 16, kernel_size=5, dilation=1, padding=1, padding_mode = 'reflect')
        self.conv1_2 = nn.Conv2d(inp_channels, 16, kernel_size=5, dilation=2, padding=3, padding_mode = 'reflect')
        self.conv1_3 = nn.Conv2d(inp_channels, 32, kernel_size=5, dilation=5, padding=9, padding_mode = 'reflect')
        self.bn_1    = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 16, kernel_size=5, dilation=1, padding=1, padding_mode = 'reflect')
        self.conv2_2 = nn.Conv2d(64, 16, kernel_size=5, dilation=2, padding=3, padding_mode = 'reflect')
        self.conv2_3 = nn.Conv2d(64, 32, kernel_size=5, dilation=5, padding=9, padding_mode = 'reflect')
        self.bn_2    = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 32, kernel_size=5, dilation=1, padding=1, padding_mode = 'reflect')
        self.conv3_2 = nn.Conv2d(64, 32, kernel_size=5, dilation=2, padding=3, padding_mode = 'reflect')
        self.conv3_3 = nn.Conv2d(64, 64, kernel_size=5, dilation=5, padding=9, padding_mode = 'reflect')
        self.bn_3    = nn.BatchNorm2d(128)

        # Bottleneck layers
        self.conv4_1 = nn.Conv2d(128, 64, kernel_size=5, dilation=1, padding=1, padding_mode = 'reflect')
        self.conv4_2 = nn.Conv2d(128, 64, kernel_size=5, dilation=2, padding=3, padding_mode = 'reflect')
        self.conv4_3 = nn.Conv2d(128, 128, kernel_size=5, dilation=5, padding=9, padding_mode = 'reflect')
        self.bn_4    = nn.BatchNorm2d(256)

        # Upsampling layers
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv3_1 = nn.Conv2d(384, 32, kernel_size=5, dilation=2, padding=3, padding_mode = 'reflect')
        self.deconv3_2 = nn.Conv2d(384, 32, kernel_size=5, dilation=5, padding=9, padding_mode = 'reflect')
        self.deconv3_3 = nn.Conv2d(384, 64, kernel_size=5, dilation=1, padding=1, padding_mode = 'reflect')
        self.bn_5    = nn.BatchNorm2d(128)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv2_1 = nn.Conv2d(192, 16, kernel_size=5, dilation=2, padding=3, padding_mode = 'reflect')
        self.deconv2_2 = nn.Conv2d(192, 16, kernel_size=5, dilation=5, padding=9, padding_mode = 'reflect')
        self.deconv2_3 = nn.Conv2d(192, 32, kernel_size=5, dilation=1, padding=1, padding_mode = 'reflect')
        self.bn_6    = nn.BatchNorm2d(64)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv1_1 = nn.Conv2d(128, 16, kernel_size=5, dilation=2, padding=3, padding_mode = 'reflect')
        self.deconv1_2 = nn.Conv2d(128, 16, kernel_size=5, dilation=5, padding=9, padding_mode = 'reflect')
        self.deconv1_3 = nn.Conv2d(128, 32, kernel_size=5, dilation=1, padding=1, padding_mode = 'reflect')
        self.bn_7    = nn.BatchNorm2d(64)

        self.final_deconv = nn.Conv2d(1 + 64, 8, kernel_size=3, padding=1, padding_mode = 'reflect')
        self.output_layer = nn.Conv2d(8, out_channels, kernel_size=3, padding=1, padding_mode = 'reflect')

    def forward(self,x0):

        # Downsampling path
        x1 = torch.cat((F.elu(self.conv1_1(x0)), F.elu(self.conv1_2(x0)), F.elu(self.conv1_3(x0))), dim=1)
        x1 = self.bn_1(x1)
        x1_pooled = F.max_pool2d(x1, kernel_size=2)
        # print(x1_pooled.shape)  

        x2 = torch.cat((F.elu(self.conv2_1(x1_pooled)), F.elu(self.conv2_2(x1_pooled)), F.elu(self.conv2_3(x1_pooled))), dim=1)
        x2 = self.bn_2(x2)
        x2_pooled = F.max_pool2d(x2, kernel_size=2)
        # print(x2_pooled.shape)  
        
        x3 = torch.cat((F.elu(self.conv3_1(x2_pooled)), F.elu(self.conv3_2(x2_pooled)), F.elu(self.conv3_3(x2_pooled))), dim=1)
        x3 = self.bn_3(x3)  
        x3_pooled = F.max_pool2d(x3, kernel_size=2)
        # print(x3_pooled.shape)

        # Bottleneck
        x4 = torch.cat((F.elu(self.conv4_1(x3_pooled)), F.elu(self.conv4_2(x3_pooled)), F.elu(self.conv4_3(x3_pooled))), dim=1)
        x4 = self.bn_4(x4)

        # Upsampling path
        y3 = self.upsample3(x4)
        y3 = F.pad(y3, (2, 2, 2, 2), mode = 'replicate')
        # print(y3.shape, x3.shape)   
        y3 = torch.cat((y3, x3), dim=1)
        y3 = torch.cat((F.elu(self.deconv3_1(y3)), F.elu(self.deconv3_2(y3)), F.elu(self.deconv3_3(y3))), dim=1)
        y3 = self.bn_5(y3)

        y2 = self.upsample2(y3)
        y2 = F.pad(y2, (5, 4, 5, 4), mode = 'replicate')
        # print(y2.shape, x2.shape)
        y2 = torch.cat((y2, x2), dim=1)
        y2 = torch.cat((F.elu(self.deconv2_1(y2)), F.elu(self.deconv2_2(y2)), F.elu(self.deconv2_3(y2))), dim=1)
        y2 = self.bn_6(y2)

        y1 = self.upsample1(y2)
        y1 = F.pad(y1, (4, 4, 4, 4), mode = 'replicate')
        # print(y1.shape, x1.shape)
        y1 = torch.cat((y1, x1), dim=1)
        y1 = torch.cat((F.elu(self.deconv1_1(y1)), F.elu(self.deconv1_2(y1)), F.elu(self.deconv1_3(y1))), dim=1)
        y1 = self.bn_7(y1)

        # print(y1.shape, x0.shape)
        y1 = F.pad(y1, (2, 2, 2, 2), mode = 'replicate')
        y0 = torch.cat((y1, x0), dim=1)
        y5 = F.elu(self.final_deconv(y0))
        y = torch.special.expit(self.output_layer(y5))
        prob = self.scaleDensity(y)

        return prob


    def scaleDensity(self, inp):
        b, c, h, w = inp.shape
        hw = h*w
        tar_den = self.tar_den
        curr_den = torch.norm(inp, p = 1, dim = (2, 3)).view(b, c, 1, 1) / (hw)
        return torch.where(curr_den > tar_den, inp / (curr_den + 1e-8) * tar_den, inp)
