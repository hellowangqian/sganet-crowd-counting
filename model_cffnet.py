import torch.nn as nn
import torch
from torchvision import models
import pdb
# counting with focus for free
class CFFNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CFFNet, self).__init__()
        self.conv1 = conv_bn_relu(in_channels=3, out_channels=16, kernel_size=7, stride=1, dilation=1, padding=3)
        self.res_block1 = bottleneck_block(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, padding=1)

        self.res_block2  = bottleneck_block(in_channels=16, out_channels=32, kernel_size=3, stride=2, dilation=1, padding=1)
        # level 3
        self.res_block3 = bottleneck_block(in_channels=32, out_channels=64, kernel_size=3, stride=2, dilation=1, padding=1)
        self.res_block3_1 = bottleneck_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        # level 4
        self.res_block4 = bottleneck_block(in_channels=64, out_channels=96, kernel_size=3, stride=2, dilation=1, padding=1)
        self.res_block4_1 = bottleneck_block(in_channels=96, out_channels=96, kernel_size=3, stride=1, dilation=1, padding=1)
        # level 5
        self.res_block5 = bottleneck_block(in_channels=96, out_channels=96, kernel_size=3, stride=1, dilation=2, padding=2)
        self.res_block5_1 = bottleneck_block(in_channels=96, out_channels=96, kernel_size=3, stride=1, dilation=2, padding=2)
        # level 6
        self.res_block6 = bottleneck_block(in_channels=96, out_channels=96, kernel_size=3, stride=1, dilation=4, padding=4)
        self.res_block6_1 = bottleneck_block(in_channels=96, out_channels=96, kernel_size=3, stride=1, dilation=4, padding=4)
        # level 7
        self.res_block7 = bottleneck_block(in_channels=96, out_channels=96, kernel_size=3, stride=1, dilation=2, padding=2)
        self.res_block7_1 = bottleneck_block(in_channels=96, out_channels=96, kernel_size=3, stride=1, dilation=2, padding=2)
        # level 8
        self.res_block8 = bottleneck_block(in_channels=96, out_channels=96, kernel_size=3, stride=1, dilation=1, padding=1)
        self.res_block8_1 = bottleneck_block(in_channels=96, out_channels=96, kernel_size=3, stride=1, dilation=1, padding=1)
        # level 9
        self.res_block9 = conv_bn_relu_x2(in_channels=384, out_channels=96, kernel_size=3, stride=1, dilation=1,padding=1)
        self.res_block9_1 = conv_bn_relu_x2(in_channels=96, out_channels=96, kernel_size=3, stride=1, dilation=1,padding=1)
        # level 10
        self.res_block10 = deconv_bn_relu(in_channels=96, out_channels=64, kernel_size=4, stride=2, dilation=1,padding=1)
        self.res_block10_1 = conv_bn_relu_x2(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=1,padding=1)
        # level 11
        self.res_block11 = deconv_bn_relu(in_channels=64, out_channels=32, kernel_size=4, stride=2, dilation=1,padding=1)
        self.res_block11_1 = conv_bn_relu_x2(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=1,padding=1)
        # level 12
        self.res_block12 = deconv_bn_relu(in_channels=32, out_channels=16, kernel_size=4, stride=2, dilation=1,padding=1)
        self.res_block12_1 = conv_bn_relu(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1,padding=1)
       
        self.denMap = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)

        x1 = self.res_block1(x)
        x2 = self.res_block2(x1)
        x3 = self.res_block3(x2)
        x3 = self.res_block3_1(x3)
        x4 = self.res_block4(x3)
        x4 = self.res_block4_1(x4)
        x5 = self.res_block5(x4)
        x5 = self.res_block5_1(x5)
        x6 = self.res_block6(x5)
        x6 = self.res_block6_1(x6)
        x7 = self.res_block7(x6)
        x7 = self.res_block7_1(x7)
        x8 = self.res_block8(x7)
        x8 = self.res_block8_1(x8)
        x = torch.cat((x4,x5,x7,x8),dim=1)
        x9 = self.res_block9(x)
        x9 = self.res_block9_1(x9)
        x10 = self.res_block10(x9)
        x10 = self.res_block10_1(x10)
        x11 = self.res_block11(x10)
        x11 = self.res_block11_1(x11)
        x12 = self.res_block12(x11)
        x12 = self.res_block12_1(x12)
        out = self.denMap(x12)
        out = self.relu(out)
        out = out.view(-1, out.shape[2], out.shape[3])
        return out
            
                
class bottleneck_block(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=1, dilation=1, padding=1):
        super(bottleneck_block,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.layer_conv1 = conv_bn_relu(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding)
        self.layer_conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=kernel_size,stride=1,dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.layer_conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1,stride=stride,dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
  
    def forward(self,x):
        out = self.layer_conv1(x)
        out = self.layer_conv2(out)
        out1 = self.bn(out)
        if self.in_channels == self.out_channels and self.stride==1:
            res = out1+x
        else:
            out2 = self.layer_conv3(x)
            out2 = self.bn1(out2)
            res = out1+out2
        return res

def conv_bn_relu(in_channels=None, out_channels=None, kernel_size=3, stride=1, dilation=1, padding=1):
    layers = []
    layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.ReLU()]
    return nn.Sequential(*layers)

def conv_bn_relu_x2(in_channels=None, out_channels=None, kernel_size=3, stride=1, dilation=1, padding=1):
    layers = []
    layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,dilation=dilation,padding=padding)]
    layers += [nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,dilation=dilation,padding=padding)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.ReLU()]
    return nn.Sequential(*layers)

def deconv_bn_relu(in_channels=None, out_channels=None, kernel_size=3, stride=1, dilation=1,padding=1):
    layers = []
    layers += [nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,dilation=dilation,padding=padding)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.ReLU()]
    return nn.Sequential(*layers)
