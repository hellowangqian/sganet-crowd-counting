#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils2 import initialize_weights
import pdb


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn=False, **kwargs):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=not self.use_bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.tconv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)

class SAModule_Head(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule_Head, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=1)
        self.branch3x3 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=3, padding=1)
        self.branch5x5 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=5, padding=2)
        self.branch7x7 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=7, padding=3)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out

class SAModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=1)
        self.branch3x3 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=3, padding=1),
                        )
        self.branch5x5 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=5, padding=2),
                        )
        self.branch7x7 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=7, padding=3),
                        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out

class SAModule2(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule2, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=1)
        self.branch3x3 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=3, dilation=2, padding=2),
                        )
        self.branch5x5 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=5, dilation=2, padding=4),
                        )
        self.branch7x7 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=7, dilation=2, padding=6),
                        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out
class SAModule3(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule3, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=1)
        self.branch3x3 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=3, dilation=4, padding=4),
                        )
        self.branch5x5 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=5, dilation=4, padding=8),
                        )
        self.branch7x7 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=7, dilation=4, padding=12),
                        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out

class DecodeModule(nn.Module):
    def __init__(self, in_channels_left, in_channels_right, use_bn):
        super(DecodeModule, self).__init__()
        out_channels_left = in_channels_left
        out_channels_right = in_channels_right//2
        self.conv_left = BasicConv(in_channels_left, out_channels_left, use_bn=use_bn,
                            kernel_size=1)
        self.conv_right = BasicDeconv(in_channels_right, out_channels_right, use_bn=use_bn,
                            kernel_size=3, padding=1)
        self.conv_bottom = BasicConv(out_channels_left+out_channels_right, out_channels_left,
                            use_bn=use_bn, kernel_size=1)
    
    def forward(self, input_left, input_right):
        left = self.conv_left(input_left)
        right = self.conv_right(input_right)
        x = torch.cat([left,right],1)
        out = self.conv_bottom(x)
        return out

class UpsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(UpsampleModule, self).__init__()
        
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = BasicConv(in_channels, out_channels, use_bn=use_bn,
                            kernel_size=3, padding=1)
        self.conv2 = BasicConv(out_channels, out_channels, use_bn=use_bn, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.upsample(x)
        out = self.conv2(x)
        return out

class TEDNet(nn.Module):
    def __init__(self, gray_input=False, use_bn=True):
        super(TEDNet, self).__init__()
        if gray_input:
            in_channels = 1
        else:
            in_channels = 3

        self.block1 = SAModule_Head(in_channels, 16, use_bn)
        self.maxPooling = nn.MaxPool2d(2, 2)
        self.block2 = SAModule(16, 32, use_bn)
        self.block3 = SAModule(32, 32, use_bn)
        self.block4 = SAModule(32, 64, use_bn)
        self.block5 = SAModule(64, 64, use_bn)
        self.block6 = SAModule(64, 128, use_bn)
        self.block7 = SAModule(128, 128, use_bn)
        self.block8 = SAModule2(128, 256, use_bn)
        self.block9 = SAModule3(256, 256, use_bn)

        self.deBlock22 = DecodeModule(32,64,use_bn) # 2nd column, 1st
        self.deBlock23 = DecodeModule(64,128,use_bn)
        self.deBlock24 = DecodeModule(128,256,use_bn)
        self.deBlock33 = DecodeModule(32,64,use_bn)
        self.deBlock34 = DecodeModule(64,128,use_bn)
        self.deBlock44 = DecodeModule(32,64,use_bn) # 4th column, 3rd
        self.upsample = UpsampleModule(32,32,use_bn)
  
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.relu = nn.ReLU()
        initialize_weights(self.modules())

    def forward(self, x):
        x = self.block1(x)
        x = self.maxPooling(x)
        x = self.block2(x)
        x3 = self.block3(x)
        x3 = self.maxPooling(x3)
        x = self.block4(x3)
        x5 = self.block5(x)
        x = self.block6(x5)
        x7 = self.block7(x)
        x = self.block8(x7)
        x9 = self.block9(x)
        x22 = self.deBlock22(x3,x5)
        x23 = self.deBlock23(x5,x7)
        x24 = self.deBlock24(x7,x9)
        x33 = self.deBlock33(x22,x23)
        x34 = self.deBlock34(x23,x24)
        x44 = self.deBlock44(x33,x34)
        z = self.upsample(x44)
        z = self.conv(z)
        z = self.relu(z)
        z = z.view(-1,z.shape[2],z.shape[3])
        return z
