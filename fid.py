from dataset import BasicDataset
from skimage.io import imsave
import os
from shutil import rmtree

import numpy as np
import torch

import hashlib
import argparse

import sys
sys.path.append("./pytorch_fid/")

from utils import get_models, nullcontext

from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3


TEMP_DIR = 'dump'
STATISTICS_DIR = 'fid_stats'
BATCH_SIZE = 20
DIMS = 768


def get_tag(string,dims):
    
    tag = hashlib.md5(string.encode()).hexdigest()
    return f'{dims}_{tag}'

def get_data_statistics(data_path,model,dims=DIMS,cuda=True,batch_size=BATCH_SIZE):

    # Create temporary dump directory
    
    tag = get_tag(data_path,dims)
    fname = os.path.join(STATISTICS_DIR,tag+'.npz')
    
    # Check if statistics exist
    
    if os.path.exists(fname):
        
        stats = np.load(fname)
        
        m, s, num = stats['m'], stats['s'], stats['num_samples']
        
    else:
        temp_path = os.path.join(TEMP_DIR,tag) + '/'
    
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
            
        if cuda:
            model.cuda()
        
        m,s = fid_score._compute_statistics_of_path(temp_path, model, batch_size=batch_size, cuda=cuda, dims=dims)

        # Remove dump folder
        rmtree(temp_path)
        
        np.savez(fname, m = m, s = s, num_samples = num)
        
    return m,s,num
    
    
def get_gen_statistics(gen,name,num_samples,model,dims=DIMS,cuda=True,batch_size=BATCH_SIZE):

    tag = get_tag(name,dims)    
    temp_path = os.path.join(TEMP_DIR,tag) + '/'

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
            
    if cuda:
        model.cuda()
    
    m,s = fid_score._compute_statistics_of_path(temp_path, model, batch_size=batch_size, cuda=cuda, dims=dims)

    # Remove dump dir
    rmtree(temp_path)
    
    return m,s


def get_fid_score(gen,gen_name,data_path,model,num_samples=-1,**kwargs):

    # Load precalculated statistics
    m1,s1,num_data = get_data_statistics(data_path,model,**kwargs)
        
    if num_samples < 0:
        num_samples = num_data

    # Calculate generator statistics
    m2,s2 = get_gen_statistics(gen,gen_name,num_samples,model,**kwargs)
    
    # Calculate FID Score
    fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value
    

class fid_scorer():
    
    def __init__(self,
                 device,
                 dims=DIMS,
                 num_samples=-1,
                 verbose=True,
                 ):
        
        self.dims = dims
        self.data_path = None
        self.gen_name = None
        self.num_samples = num_samples
        self.cuda = device.type == 'cuda'
        
        if verbose:
            print('Loading inception model')

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception = InceptionV3([block_idx]).to(device)
        #self.inception = torch.nn.Module()
        
        if verbose:
            print('Finished loading inception')
        
    def set_params(self,data_path,gen_name,writer):
        
        self.writer = writer
        self.data_path = data_path
        self.gen_name = gen_name
        
    def get_score(self,gen,step):
        
        assert self.data_path and self.gen_name, "run set_params()"
                
        fid_score = get_fid_score(gen,self.gen_name,self.data_path,self.inception,
                                num_samples=self.num_samples,cuda=self.cuda)
        
        if self.writer:
            self.writer.add_scalar('FID_score',fid_score,step)
        
        return fid_score


if __name__ == '__main__':
    
    print('Loading hyperparameters.')
    
    # Get location of hparams.txt
    parser = argparse.ArgumentParser(description='Train a GAN!')
    parser.add_argument('-H', '--hparams', metavar='filename', type=str, default='hparams.txt',
                        help='File containing hyperparamters', dest='hparams')
    parser.add_argument('-g', '--gpu', metavar='number', type=int, default='0',
                        help='Number of GPU to use', dest='gpu')
    parser.add_argument('-d', '--dims', metavar='number', type=int, default=DIMS, 
                        help='Number of dimensions of inception layer', dest='dims',
                        choices=[64,192,768,2048])
    parser.add_argument('-s', '--samples', metavar='number', type=int, default=-1, 
                        help='Number of samples from generator', dest='samples')
    args = parser.parse_args()

    # Set device
    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    print(f'Using device: {device}')
    
    
    print('Loading inception model')
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    inception = InceptionV3([block_idx]).to(device)
    
    print('Finished loading inception')
    
    with torch.cuda.device(device) if cuda else nullcontext:
        
        # Load params from text file
        models = get_models(args.hparams,device,load_discr=False)
                           
        print('Entering Hyperparameter Loop')
            
        for h,gen,_ in models:
    
            fid = get_fid_score(gen,h.name,h.dataroot,inception,
                                num_samples=args.samples,cuda=cuda)
            
            print(fid)
        