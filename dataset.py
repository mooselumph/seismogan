# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 08:45:25 2020

@author: mooselumph
"""

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging


class BasicDataset(Dataset):
    
    def __init__(self, model_dir=None, gather_dir=None, scale=1):
        self.model_dir = model_dir
        self.gather_dir = gather_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        d = model_dir if model_dir else gather_dir

        self.ids = [splitext(file)[0] for file in listdir(d)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, i):
        print('here')
        idx = self.ids[i]
        
        d = dict()

        if self.model_dir:
        
            model_file = glob(self.model_dir + idx + '*')            
            assert len(model_file) == 1, \
                f'Either no model or multiple models found for the ID {idx}: {model_file}'

            model = np.load(model_file[0])[np.newaxis,:,:]
            d['model'] = torch.tensor(model,dtype=torch.float32)
        
        if self.gather_dir:
            gather_file = glob(self.gather_dir + idx + '*')
    
            assert len(gather_file) == 1, \
                f'Either no gather file or multiple files found for the ID {idx}: {gather_file}'
                
            d['gather'] = torch.tensor(np.load(gather_file[0]),dtype=torch.float32)

        return d 

