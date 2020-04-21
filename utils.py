# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:15:16 2020

@author: mooselumph
"""

import os, glob, re
import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace

from models import DCGAN, DCGAN_SN, original, original_SN, original_SN2

def load_hparams(fname,defaults):
    
    
    htable = pd.read_table(fname,sep='\s+')
    
    assert 'name' in htable.columns.values, 'name must be specified'

    args = []

    for i in range(htable.shape[0]):
        
        A = dict(htable.iloc[i,:])
                
        for key in A:
            if type(A[key]) == np.int64:
                A[key] = int(A[key])
                
        B = defaults.copy()
        B.update(A)
                
        h = SimpleNamespace(**B)
        
        args.append(h)
        
    return args


def get_models(fname_hparams,device,load_gen=True,load_discr=True,verbose=True):
    
     # Set default params
    defaults = {
        'model': 'original',
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
        'image_interval': 20,
        'save_interval': 20,
        'score_interval': 20,
        'dataroot': '/home/raynor/datasets/april/velocity/',
        'modelroot': '/home/raynor/code/seismogan/saved/',
        'load_name': 'None',
        'load_step': -1,
        }
    
    # Load params from text file
    hparams = load_hparams(fname_hparams,defaults)
                          
    for i,h in enumerate(hparams):

            if os.path.exists(os.path.join(h.modelroot,h.name)):
                print(f'{h.name} folder exists. Skipping.')
                continue
        
            if verbose:
                print('Loading models')
            
            if h.model == 'DCGAN':
                gen = DCGAN.Generator(h.nz, h.nc, h.ngf, device) if load_gen else None
                discr = DCGAN.Discriminator(h.nc, h.ndf, device) if load_discr else None
            elif h.model == 'DCGAN_SN':
                gen = DCGAN_SN.Generator(h.nz, h.nc, h.ngf, device) if load_gen else None
                discr = DCGAN_SN.Discriminator(h.nc, h.ndf, device) if load_discr else None
            elif h.model == 'original':
                gen = original.Generator(h.nz, h.nc, h.ngf, device) if load_gen else None
                discr = original.Discriminator(h.nc, h.ndf, device) if load_discr else None
            elif h.model == 'original_SN':
                gen = original_SN.Generator(h.nz, h.nc, h.ngf, device) if load_gen else None
                discr = original_SN.Discriminator(h.nc, h.ndf, device) if load_discr else None
            elif h.model == 'original_SN2':
                gen = original_SN2.Generator(h.nz, h.nc, h.ngf, device) if load_gen else None
                discr = original_SN2.Discriminator(h.nc, h.ndf, device) if load_discr else None
            else:
                raise NotImplementedError
            
            if h.load_name.lower() != 'none':
                fname = load_models(os.path.join(h.modelroot,h.load_name),gen,discr,h.load_step)
                
                if verbose:
                    print (f'Loaded model: {fname}')
                    
            h.has_next = i+1 < len(hparams)
                    
            yield h,gen,discr





def save_models(save_dir,gen,discr,step):
    
    name = f'checkpoint_{step}.pth'
    
    torch.save({
        'gen': gen.state_dict(),
        'discr': discr.state_dict()
        }, os.path.join(save_dir,name))
    
def load_models(load_dir,gen,discr,step=None):
    
    if step != None and step >= 0:
        name = f'checkpoint_{step}.pth'
    else:
        files = glob.glob(os.path.join(load_dir,"*.pth"))
        files = [os.path.splitext(os.path.basename(f))[0] for f in files]
        steps = [int(re.findall('checkpoint_(.+)',f)[0]) for f in files]
        
        assert steps, "No models of the specified name were found."
        
        step = max(steps)
        name = f'checkpoint_{step}.pth'
        
    fname =  os.path.join(load_dir,name)   
    checkpoint = torch.load(fname)
    
    if discr:
        discr.load_state_dict(checkpoint['discr'])
    if gen:
        gen.load_state_dict(checkpoint['gen'])

    return fname


class nullcontext():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False
    def __bool__(self):
        return False