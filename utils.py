# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:15:16 2020

@author: mooselumph
"""
import numpy as np
import pandas as pd
from types import SimpleNamespace

def load_hparams(fname,defaults):
    
    
    htable = pd.read_table(fname,sep='\t+',engine='python')
    
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
