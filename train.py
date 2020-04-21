# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:50:01 2020

@author: mooselumph
"""

import random
import numpy as np
import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from utils import get_models, save_models, nullcontext

from dataset import BasicDataset

from fid import fid_scorer


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
        device,
        n_epochs = 500,
        n_discr = 1,
        n_gen = 1,
        image_interval = 20,
        save_interval = 20,
        score_interval = 20,
        save_dir = None,
        tb_writer = None,
        scheduler = None,
        scorer = None,
        seed = 999,
        verbose = True,
        ):
    
    
    # Set random seed for reproduceability
    random.seed(seed)
    torch.manual_seed(seed)
            
    step = 0
    
    try:
        
        with tqdm(total=n_epochs, desc=f'Progress: ',unit='epoch') as pbar:            
    
            for epoch in range(n_epochs):
                                
                for real in dataloader:
                    
                    # Get real and fake data
                    real = real.to(device)
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
                        
                    is_last_epoch = (step == len(dataloader)*n_epochs - 1)
                    
                    # Log Losses and Images
                    stats = {'loss_D':loss_discr.item(),'loss_G':loss_discr.item()}
                    if tb_writer:
                        tb_writer.add_scalar('D_real',torch.sigmoid(D_real).mean().item(),step)
                        tb_writer.add_scalar('D_fake',torch.sigmoid(D_fake).mean().item(),step)
                        tb_writer.add_scalar('loss_D',stats['loss_D'],step)
                        tb_writer.add_scalar('loss_G',stats['loss_G'],step)
                    
                        # Show generated images
                        if step % int(image_interval*len(dataloader)) == 1 or is_last_epoch:
                            
                            with torch.no_grad():
                                fake = gen.fixed_sample().detach().cpu()
                            
                            fake = F.interpolate(fake,scale_factor=[1,1,0.5,0.5],mode='bilinear')
                            fake = vutils.make_grid(fake, padding=2, normalize=True)[np.newaxis]
                            tb_writer.add_images('images', fake, step)
                        
                    # Save model
                    if step % int(save_interval*len(dataloader)) == 1 or is_last_epoch:
                        save_models(save_dir,gen,discr,step)
                        
                    # Get score
                    if scorer:
                        if step % int(score_interval*len(dataloader)) == 1 or is_last_epoch:
                            if verbose:
                                print('Calculating score')
                            stats['score'] = scorer.get_score(gen,step)
                        
                    step += 1
            
                # Update progress bar
                pbar.set_postfix(**stats)
                pbar.update()
            
        return True
                
    except KeyboardInterrupt: 
        save_models(save_dir,gen,discr,step)
        return False

            
if __name__ == '__main__':
    
    
    print('Loading hyperparameters')
    
    # Get location of hparams.txt
    parser = argparse.ArgumentParser(description='Train a GAN!')
    parser.add_argument('-H', '--hparams', metavar='filename', type=str, default='hparams.txt',
                        help='File containing hyperparamters', dest='hparams')
    parser.add_argument('--gpu', metavar='number', type=int, default=0,
                        help='Number of GPU to use', dest='gpu')
    parser.add_argument('--use_writer', type=int, default=1,
                        choices=[0,1],help='Whether to write to tensorbaord')                    
    parser.add_argument('--use_scorer', type=int, default=1,
                        choices=[0,1],help='Whether to calculate FID score')    
    args = parser.parse_args()


    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available()) else "cpu")
    print(f'Using device: {device}')
    
    # Load scorer
    scorer = fid_scorer(device) if args.use_scorer else None
    
    # Load params from text file
    models = get_models(args.hparams,device)
                       
    print('Entering Hyperparameter Loop')
        
    for h,gen,discr in models:
        
        with (SummaryWriter(comment=f'_{h.name}') if args.use_writer else nullcontext) as writer:
            
            print(f'Run name: {h.name}')

            if writer:
                writer.add_hparams(vars(h),{})
            
            if scorer:
                scorer.set_params(h.dataroot,h.name,writer)
                    
            dataset = BasicDataset(model_dir=h.dataroot)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=h.batch_size, num_workers=8, shuffle=True)
    
            gen_opt = optim.Adam(gen.parameters(), lr=h.lrG, betas=(h.beta1, h.beta2))
            discr_opt = optim.Adam(discr.parameters(), lr=h.lrD, betas=(h.beta1, h.beta2))
        
            # Create save dir
            save_dir = os.path.join(h.modelroot,h.name)        
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                
            print('Beginning training', flush=True)
        
            completed = train(
                dataloader,
                discr,
                gen,
                minimax_discr_loss,
                minimax_gen_loss,
                discr_opt,
                gen_opt,
                device,
                n_epochs = h.n_epochs,
                n_discr = h.nD,
                n_gen = h.nG,
                image_interval = h.image_interval,
                save_interval = h.save_interval,
                score_interval = h.score_interval,
                save_dir = save_dir,
                tb_writer = writer,
                scorer = scorer
                  )
            
            if completed:
                print('Completed training')
            
            else:
                print('Interrupted models saved.')
                if h.has_next:
                     response = input("Would you like to exit the hyperparameter loop? (y/n):\n")
                     if response == 'y':
                         break
                
                
                
    
        
        
        
