# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:05:03 2020

@author: mooselumph
"""


import torch.utils.data
from dataset import BasicDataset

dataroot = 'C:/Users/mooselumph/code/data/velocity/'
dataset = BasicDataset(model_dir=dataroot)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=10,
                                         shuffle=False,num_workers=1)

for batch in dataloader:
    print(batch.shape)