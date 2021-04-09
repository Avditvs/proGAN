import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MinibatchStdLayer(nn.Module):

    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size
        
    def forward(self, x):
        size = x.size()
        subGroupSize = min(size[0], self.group_size)
        if size[0] % subGroupSize != 0:
            subGroupSize = size[0]
        G = int(size[0] / subGroupSize)
        if subGroupSize > 1:
            y = x.view(-1, subGroupSize, size[1], size[2], size[3])
            y = torch.var(y, 1)
            y = torch.sqrt(y + 1e-8)
            y = y.view(G, -1)
            y = torch.mean(y, 1).view(G, 1)
            y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
            y = y.expand(G, subGroupSize, -1, -1, -1)
            y = y.contiguous().view((-1, 1, size[2], size[3]))
        else:
            y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
    
        return torch.cat([x, y], dim=1)

class WeightedConv2d(nn.Module):
    def __init__(self, in_filters, out_filters, size, strides, padding, gain = np.sqrt(2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=size, stride=strides, padding=padding)
        
        
        # new bias to use after wscale
        bias = self.conv.bias
        self.bias = nn.Parameter(bias.view(1, bias.shape[0], 1, 1))
        self.conv.bias = None
        
        # calculate scale
        convShape = list(self.conv.weight.shape)
        fanIn = np.prod(convShape[1:]) # Leave out # of op filters
        self.wtScale = gain/np.sqrt(fanIn)
        
        # init
        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.bias, val=0)
        self.name = '(inp = %s)' % (self.conv.__class__.__name__ + str(convShape))
        
    def forward(self, x):
        #return self.conv(x)
        return self.conv(x * self.wtScale) + self.bias

class PixelwiseNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        factor = ((x**2).mean(dim=1, keepdim=True) + 1e-8).sqrt()
        return x / factor

class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=2)

class WeightedLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bias = self.linear.bias
        self.linear.bias = None
        fanIn = in_dim
        self.wtScale = np.sqrt(2) / np.sqrt(fanIn)

        nn.init.normal_(self.linear.weight)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x):
        return self.linear(x * self.wtScale) + self.bias


class DiscrConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, size, strides, padding, ):
        super().__init__()
        self.core = nn.Sequential(
            WeightedConv2d(in_filters, out_filters, size, strides, padding),
            nn.LeakyReLU(0.2)          
        )
        
    def forward(self, x):
        return self.core(x)
        
class DiscrBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.core = nn.Sequential(
            DiscrConvBlock(in_filters, out_filters, 3, 1, 1),
            DiscrConvBlock(out_filters, out_filters, 3, 1, 1),
            nn.AvgPool2d(2),
        )
        
    def forward(self, x):
        return self.core(x)

class DiscrInitBlock(nn.Module):
    def __init__(self, out_filters, input_dim):
        super().__init__()
        self.core = DiscrConvBlock(input_dim, out_filters, 1, 1, 0)
        
    def forward(self, x):
        return self.core(x)

class DiscrFinalBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.out_filters = out_filters
        self.convs = nn.Sequential(
            DiscrConvBlock(in_filters+1, out_filters, 3, 1, 1),
            DiscrConvBlock(out_filters, out_filters, 4, 1, 0),
        )
        
        self.std = MinibatchStdLayer()
        
        self.linear = WeightedLinear(out_filters, 1)
        
    def forward(self, x):
        x = self.std(x)
        x = self.convs(x)
        x = x.view(-1, self.out_filters)
        return self.linear(x)

class GenConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, size, strides, padding):
        super().__init__()
        self.core = nn.Sequential(
            WeightedConv2d(in_filters, out_filters, size, strides, padding),
            nn.LeakyReLU(0.2),
            PixelwiseNorm(),
        )
        
    def forward(self, x):
        return self.core(x)

class GenBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.core = nn.Sequential(
            Upsample(),
            GenConvBlock(in_filters, out_filters, 3, 1, 1),
            GenConvBlock(out_filters, out_filters, 3, 1, 1),
        )
    
    def forward(self, x):
        return self.core(x)

class GenInitBlock(nn.Module):
    def __init__(self, out_filters, latent_dim):
        super().__init__()
        self.core = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, out_filters, 4, 1, 0),
                nn.LeakyReLU(0.2),
                PixelwiseNorm(),
                GenConvBlock(out_filters, out_filters, 3, 1, 1),
            )
        
    def forward(self, x):
        return self.core(x)

class GenFinalBlock(nn.Module):
    def __init__(self, in_filters, out_dim):
        super().__init__()
        self.core = nn.Sequential(
            WeightedConv2d(in_filters, out_dim, 1, 1, 0),
        )
        
    def forward(self, x):
        return self.core(x)

def update_ema(dest, src, beta):
    with torch.no_grad():
        src_param_dict = dict(src.named_parameters())

        for p_name, p_dest in dest.named_parameters():
            p_src = src_param_dict[p_name]
            assert p_src is not p_dest
            p_dest.copy_(beta * p_dest + (1.0 - beta) * p_src)
