# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:50:01 2020

@author: mooselumph
"""

import random
import numpy as np
import os, sys
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace

from utils import load_hparams, save_models, load_models

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
        image_interval = 1,
        save_interval = 1,
        save_dir = None,
        scheduler = None,
        seed = 999,
        ):
    
    
    random.seed(seed)
    torch.manual_seed(seed)
            
    step = 0
    
    try:
    
        for epoch in range(n_epochs):
            
            for real in dataloader:
                
                # Get real and fake data
                batch_size = real.shape[0]
                fake = gen.sample(batch_size = batch_size)
                                
                # Train Discriminator
                discr_optimizer.zero_grad()  

                D_real = discr(real)
                D_fake = discr(fake.detach())

                loss_discr = discr_loss(D_real,D_fake)
                loss_discr.backward()
    
                if epoch % (n_gen + n_discr) < n_discr:
                    discr_optimizer.step()
                                
                # Train Generator
                gen_optimizer.zero_grad()

                D_fake = discr(fake)

                loss_gen = gen_loss(D_fake)
                loss_gen.backward()

                if epoch % (n_gen + n_discr) >= n_discr:
                    gen_optimizer.step()
                    
                # Log Losses
                tb_writer.add_scalars('Discriminator',
                                      {'D_real':torch.sigmoid(D_real).mean().item(),
                                       'D_fake':torch.sigmoid(D_fake).mean().item()},step)
                tb_writer.add_scalar('loss_D',loss_discr,step)
                tb_writer.add_scalar('loss_G',loss_gen,step)
                
                # Show generated images
                if step % int(image_interval*len(dataloader)) == 1 or (step == len(dataloader)*n_epochs - 1):
                    
                    with torch.no_grad():
                        fake = gen.fixed_sample().detach().cpu()    
                    #grid = vutils.make_grid(fake, padding=2, normalize=True[np.newaxis]
                    tb_writer.add_images('images', fake, step)
                    
                    
                if step % int(save_interval*len(dataloader)) == 1 or (step == len(dataloader)*n_epochs - 1):
                    save_models(save_dir,gen,discr,step)
                    
                    
                step += 1
        
            
        return True
                
    except KeyboardInterrupt: 
        save_models(save_dir,gen,discr,step)
        return False

            
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
        'save_interval': 2,
        'dataroot': '/home/raynor/datasets/april/velocity/',
        'modelroot': '/home/raynor/code/seismogan/saved/',
        'load_name': '',
        'load_step': -1,
        }
    
    # Load params from text file
    hparams = load_hparams(args.hparams,defaults)
    
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available()) else "cpu")
                       
    print('Entering Hyperparameter Loop')
        
    for i,h in enumerate(hparams):
        
        with SummaryWriter(comment=f'_{h.name}') as writer:
            
            writer.add_hparams(vars(h),{})
                    
            dataset = BasicDataset(model_dir=h.dataroot,device=device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=h.batch_size, shuffle=True)
        
            print('Loading models')
            
            gen = Generator(h.nz, h.nc, h.ngf, device)
            discr = Discriminator(h.nc, h.ndf, device)
            
            if h.load_name:
                load_models(os.path.join(h.modelroot,h.load_name),gen,discr,h.load_step)
                
            gen_opt = optim.Adam(gen.parameters(), lr=h.lrG, betas=(h.beta1, h.beta2))
            discr_opt = optim.Adam(discr.parameters(), lr=h.lrD, betas=(h.beta1, h.beta2))
        
            # Create save dir
            save_dir = os.path.join(h.modelroot,h.name)        
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                
            print('Beginning training')
        
            completed = train(
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
                save_interval = h.save_interval,
                save_dir = save_dir
                  )
            
            if completed:
                print('Completed training')
            
            else:
                print('Interrupted models saved.')
                if i+1 < len(hparams):
                     response = input("Would you like to exit the hyperparameter loop? (y/n):\n")
                     if response == 'y':
                         break
                
                
                
    
        
        
        
