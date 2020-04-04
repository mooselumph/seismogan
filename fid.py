from dataset import BasicDataset
from skimage.io import imsave
import os
import numpy as np
import torch

import model

import sys
sys.path.append("./pytorch_fid/")

from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3


TEMP_DIR = 'dump'
STATISTICS_DIR = 'fid_stats'
BATCH_SIZE = 20
DIMS = 768


def precalculate_data_statistics(data_path,name,dims=DIMS,cuda=True,data_statistics_path=STATISTICS_DIR,temp_path=TEMP_DIR,batch_size=BATCH_SIZE):

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

        # Load all images and convert to save as 0-255 .png or .jpg (normalize per image)
                            
        dataset = BasicDataset(model_dir=data_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

        num = 0

        for batch in dataloader:

            batch = batch.detach().cpu().numpy()
            batch = ((batch+1)/2*255).astype(np.uint8)
            
            for i in range(batch.shape[0]):
            
                img = np.tile(batch[i].transpose(1,2,0),(1,1,3)) 
                imsave(os.path.join(temp_path,f'img_{num}.png'),img)
                
                num += 1
    else:
        num = len(os.listdir(temp_path))
    
    # Get statistics for folder of images
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    
    if cuda:
        model.cuda()
    
    m,s = fid_score._compute_statistics_of_path(temp_path, model, batch_size=batch_size, cuda=cuda, dims=dims)
    
    np.savez(os.path.join(data_statistics_path,name + '.npz'),m = m, s = s, num_samples = num)
    
    
def get_gen_statistics(gen,num_samples,cuda=True,dims=DIMS,temp_path=TEMP_DIR,data_statistics_path=STATISTICS_DIR,batch_size=BATCH_SIZE):

    # Generate a folder full of generated images
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
        
        num = 0
        
        n_batches = int(num_samples / batch_size)+1
        for k in range(n_batches):
        
            batch = gen.sample(batch_size = batch_size).detach().cpu().numpy()
            batch = ((batch+1)/2*255).astype(np.uint8)
            
            for i in range(batch_size):
            
                img = np.tile(batch[i].transpose(1,2,0),(1,1,3))
                imsave(os.path.join(temp_path,f'img_{num}.png'),img)
                
                num += 1
            
    # Calculate Statistics
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    
    if cuda:
        model.cuda()
    
    m,s = fid_score._compute_statistics_of_path(temp_path, model, batch_size=batch_size, cuda=cuda, dims=dims)
    
    return m,s

def get_fid_score(gen,data_name,num_samples=-1,data_statistics_path=STATISTICS_DIR,**kwargs):

    # Load precalculated statistics
    stats = np.load(os.path.join(data_statistics_path,data_name + '.npz'))
    
    if num_samples < 0:
        num_samples = stats['num_samples']

    # Calculate generator statistics
    m2,s2 = get_gen_statistics(gen,num_samples,**kwargs)
    
    # Calculate FID Score
    fid_value = fid_score.calculate_frechet_distance(stats['m'], stats['s'], m2, s2)
    
    return fid_value
    

if __name__ == '__main__':

    saved = '/home/raynor/code/seismogan/saved/brielle1/checkpoint_18601.pth'
    checkpoint = torch.load(saved)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    gen = model.Generator(nz=100,ngf=64,nc=1,device=device)
    gen.load_state_dict(checkpoint['gen'])
    
    fid = get_fid_score(gen,'vel1',temp_path='dump/brielle1')
    
    print(fid)
    