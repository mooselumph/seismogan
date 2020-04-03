# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:50:01 2020

@author: mooselumph
"""

import numpy as np
import os, sys
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace

from utils import load_hparams

from model import Discriminator, Generator
from dataset import BasicDataset


def minimax_discr_loss(D_real,D_fake):
    """
    Assumes that discriminator output is in range (-inf,inf)
    """
    
    ones = torch.ones(D_real.shape, device=D_real.device)
    loss_real = F.binary_cross_entropy(torch.sigmoid(D_real), ones)
    
    zeros = torch.zeros(D_fake.shape, device=D_fake.device)
    loss_fake = F.binary_cross_entropy(torch.sigmoid(D_fake), zeros)
    
    return loss_real + loss_fake

def minimax_gen_loss(D_fake):
    """
    Maximizes log(D(G(z))) instead of minimizing log(1-D(G(z)))
    """
    
    ones = torch.ones(D_fake.shape, device=D_fake.device)
    loss = F.binary_cross_entropy(torch.sigmoid(D_fake), ones) 
    
    return loss


def train(  
        dataloader,
        discr,
        gen,
        discr_loss,
        gen_loss,
        discr_optimizer,
        gen_optimizer,
        tb_writer,
        n_epochs = 500,
        n_discr = 1,
        n_gen = 1,
        image_interval = 100,
        scheduler = None,
        ):
        
    step = 0
    
    for epoch in range(n_epochs):
        
        for real in dataloader:
            
            discr_optimizer.zero_grad()            
            gen_optimizer.zero_grad()
            
            # Get real and fake data
            batch_size = real.shape[0]
            fake = gen.sample(batch_size = batch_size)
                            
            # Train Discriminator
            D_real = discr(real)
            D_fake = discr(fake)
            loss_discr = discr_loss(D_real,D_fake)

            if epoch % (n_gen + n_discr) <= n_discr:
                loss_discr.backward()
                discr_optimizer.step()
                            
            # Train Generator
            D_fake = discr(fake)
            loss_gen = gen_loss(D_fake)
            
            if epoch % (n_gen + n_discr) > n_gen:
                loss_gen.backward()
                gen_optimizer.step()
                
            # Log Losses
            tb_writer.add_scalars('Discriminator',
                                  {'D_real':torch.sigmoid(D_real).mean().item(),
                                   'D_fake':torch.sigmoid(D_fake).mean().item()},step)
            tb_writer.add_scalar('loss_D',loss_discr,step)
            tb_writer.add_scalar('loss_G',loss_gen,step)
            
            # Show generated images
            if step % int(image_interval*batch_size*len(dataloader)) == 1:
                
                with torch.no_grad():
                    fake = gen.sample(batch_size= real.shape[0]).detach().cpu()    
                #grid = vutils.make_grid(fake, padding=2, normalize=True[np.newaxis]
                tb_writer.add_images('images', fake, step)
                
            step += 1
            
if __name__ == '__main__':
    
    
    print('Loading hyperparameters.')
    
    # Get location of hparams.txt
    parser = argparse.ArgumentParser(description='Train a GAN!')
    parser.add_argument('-H', '--hparams', metavar='filename', type=str, default='hparams.txt',
                        help='File containing hyperparamters', dest='hparams')
    parser.add_argument('-g', '--gpu', metavar='number', type=int, default='0',
                        help='Number of GPU to use', dest='gpu')
    args = parser.parse_args()

    # Set default params
    defaults = {
        'nz': 100,
        'nc': 1,
        'ndf': 64,
        'ngf': 64,
        'n_epochs': 500,
        'batch_size': 100,
        'lrD': 0.0001,
        'lrG': 0.0001,        
        'beta1': 0.5,
        'beta2': 0.999,
        'nD': 1,
        'nG': 2,
        'image_interval': 1,
        'dataroot': '/home/raynor/datasets/april/velocity/',
        'modelroot': '/home/raynor/code/seismogan/saved/',
        'loadD': False,
        'loadG': False,
        }
    
    # Load params from text file
    hparams = load_hparams(args.hparams,defaults)
    
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available()) else "cpu")
                       
    print('Entering Hyperparameter Loop')
        
    for i,h in enumerate(hparams):
        
        
        writer = SummaryWriter(comment=f'_{h.name}')
        writer.add_hparams(vars(h),{})
                
        dataset = BasicDataset(model_dir=h.dataroot,device=device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=h.batch_size, shuffle=True)
    
        print('Loading models')
        
        gen = Generator(h.nz, h.nc, h.ngf, device)
        discr = Discriminator(h.nc, h.ndf, device)
        
        if h.loadD:
            discr.load_state_dict(torch.load(os.path.join(h.modelroot,h.loadD)))
            
        if h.loadG:
            discr.load_state_dict(torch.load(os.path.join(h.modelroot,h.loadG)))
        
        gen_opt = optim.Adam(gen.parameters(), lr=h.lrG, betas=(h.beta1, h.beta2))
        discr_opt = optim.Adam(discr.parameters(), lr=h.lrD, betas=(h.beta1, h.beta2))
    
        print('Beginning training')
    
        try:
            train(
                dataloader,
                discr,
                gen,
                minimax_discr_loss,
                minimax_gen_loss,
                discr_opt,
                gen_opt,
                writer,
                n_epochs = h.n_epochs,
                n_discr = h.nD,
                n_gen = h.nG,
                image_interval = h.image_interval,
                  )
            
            print('Completed training')
            
            torch.save(gen.state_dict(), os.path.join(h.modelroot,f'gen_{h.name}.pth'))
            torch.save(discr.state_dict(), os.path.join(h.modelroot,f'discr_{h.name}.pth'))
            
        except KeyboardInterrupt:
            
            torch.save(gen.state_dict(), os.path.join(h.modelroot,f'gen_{h.name}_interrupted.pth'))
            torch.save(discr.state_dict(), os.path.join(h.modelroot,f'discr_{h.name}_interrupted.pth'))
            
            print('Interrupted models saved.')
            
            if i+1 < len(hparams):
                response = input("Would you like to exit the hyperparameter loop? (y/n):\n")
                if response != 'y':
                    continue
            
            sys.exit(0)
            
    
        
        
        
