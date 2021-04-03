import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import progan.custom_modules as cm

class Discriminator(nn.Module):
    def __init__(self, latent_dim, num_filters, max_filters, depth, input_dim):
        
        super(Discriminator, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.pooling = nn.AvgPool2d(2)
        self.depth = depth
        self.max_filters = max_filters
        self.rank=0
                
        self.init_blocks = nn.ModuleList([cm.DiscrInitBlock(out_filters = min(self.max_filters, self.num_filters * 2**i), input_dim = self.input_dim) for i in reversed(range(self.depth+1))])
        self.blocks = nn.ModuleList([cm.DiscrBlock(in_filters = min(self.max_filters, self.num_filters * 2**i), out_filters = min(self.max_filters, self.num_filters * 2**(i+1))) for i in reversed(range(self.depth))])    
        self.final_block = cm.DiscrFinalBlock(min(self.max_filters, self.num_filters * 2**self.depth), min(self.max_filters, self.num_filters * 2**self.depth))
        
    def grow(self, growth_factor = 1):
        self.rank += growth_factor
                
    def forward(self, x, alpha=1):
        
        if self.rank==0:
            x = self.init_blocks[self.rank](x)
        
        else:        
            old_input = self.pooling(x)
            old_input = self.init_blocks[self.rank-1](old_input)

            new_input = self.init_blocks[self.rank](x)
            new_input = self.blocks[self.rank-1](new_input)

            x = (1-alpha)*old_input+alpha*new_input
        
            for i in reversed(range(self.rank-1)):
                x = self.blocks[i](x)
                
        x = self.final_block(x)
        return x
    
    def get_device(self):
        return next(self.parameters()).device
        

class Generator(nn.Module):
    def __init__(self, latent_dim, num_filters, max_filters, depth, output_dim):
        
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.output_dim = output_dim
        self.max_filters = max_filters
        self.depth = depth
        self.rank = 0
        
        self.init_block = cm.GenInitBlock(out_filters = min(self.max_filters, self.num_filters * 2**self.depth), latent_dim=self.latent_dim)
        self.blocks = nn.ModuleList([cm.GenBlock(in_filters = min(self.max_filters, self.num_filters * 2**(i+1)),  out_filters = min(self.max_filters, self.num_filters * 2**i)) for i in reversed(range(self.depth))])
        self.final_blocks = nn.ModuleList([cm.GenFinalBlock(in_filters = min(self.max_filters, self.num_filters * 2**i), out_dim = output_dim) for i in reversed(range(self.depth+1))])
        
    def grow(self, growth_factor=1):
        self.rank += growth_factor
        
    def forward(self, x, alpha=1):
        x = self.init_block(x)
        
        if self.rank!=0:
            for i in range(self.rank-1):
                x = self.blocks[i](x)
            
            old_output = self.final_blocks[self.rank-1](x)
            
            x = self.blocks[self.rank-1](x)
            
            new_output = self.final_blocks[self.rank](x)
            final_output = (1-alpha)*F.interpolate(old_output, scale_factor=2)+alpha*new_output
        
        else:
            final_output = self.final_blocks[self.rank](x)
        
        return final_output

    def get_device(self):
        return next(self.parameters()).device
    
    def generate_latent_points(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, 1,1, device=self.get_device())