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
        step = max(steps)
        name = f'checkpoint_{step}.pth'
        
    fname =  os.path.join(load_dir,name)   
    checkpoint = torch.load(fname)
    discr.load_state_dict(checkpoint['discr'])
    gen.load_state_dict(checkpoint['gen'])

    return fname
