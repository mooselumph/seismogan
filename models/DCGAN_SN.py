# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:48:29 2020

@author: mooselumph
"""

import torch
import torch.nn as nn

from models.spectralnorm import SpectralNorm2d


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('SpectralNorm') != -1:
        nn.init.normal_(m.w_bar.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code

class Up(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=4, stride=2, padding=1):
        self.main = nn.Sequential(
            SpectralNorm2d(nn.ConvTranspose2d( in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self,input):
        return self.main(input)

class Down(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=4, stride=2, padding=1, batch_norm=True):

        self.conv = SpectralNorm2d(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self,input):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)

        return x


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, device):
        super(Generator, self).__init__()
        
        self.device = device
        self.nz = nz
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            Up(nz, ngf*16, stride=1, padding=0),
            # state size. (ngf*16) x 4 x 4
            Up(ngf*16, ngf*8),
            # state size. (ngf*8) x 8 x 8
            Up(ngf*8, ngf*4),
            # state size. (ngf*4) x 16 x 16
            Up(ngf*4, ngf*2),
            # state size. (ngf*2) x 32 x 32
            Up(ngf*2, ngf),
            # state size. (ngf) x 64 x 64 
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        
        self.to(device)
        
        self.apply(weights_init)
    
    def sample(self,batch_size):
        
        noise = torch.randn((batch_size, self.nz, 1, 1), device=self.device)
        fake = self.forward(noise)
        return fake
    
    def fixed_sample(self):
        
        return self.forward(self.fixed_noise)

    def forward(self, input):
        return self.main(input)
    
    
class Discriminator(nn.Module):
    def __init__(self, nc, ndf, device):
        super(Discriminator, self).__init__()
        
        self.device = device
        
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            Down(nc,ndf,batch_norm=False),
            # state size. (ndf) x 64 x 64
            Down(ndf, ndf*2),
            # state size. (ndf*2) x 32 x 32
            Down(ndf*2, ndf*4),
            # state size. (ndf*4) x 16 x 16
            Down(ndf*4, ndf*8),
            # state size. (ndf*8) x 8 x 8
            Down(ndf*8, ndf*16),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        )
        
        self.to(device)
                
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)
    
