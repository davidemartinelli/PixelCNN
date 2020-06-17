from torch.utils.data import Dataset
from torchvision.datasets import MNIST

import subprocess
import numpy as np

class BinaryMNIST(Dataset):
    def __init__(self, train=True):
        self.load_data(train)
        
    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image[image < 127] = 0
        image[image >= 127] = 255 #leave 255, otherwise if you retrieve an image twice the second time it will be all zeros

        return image.unsqueeze(0) / 255., label

    def __len__(self):
        return len(self.images)
    
    def load_data(self, train):
        dataset = MNIST(root='./data', train=train, download=True)
        
        self.images , self.labels = dataset.data, dataset.targets

def get_gpu_memory_map():
    '''
    This function returns the ID of the GPU 
    with current lowest memory usage.
    '''

    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    result = result.decode('utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    
    #determine the lowest memory gpu
    minimum_memory = np.inf
    best_gpu = None
    for key in gpu_memory_map:
        if gpu_memory_map[key] < minimum_memory:
            minimum_memory = gpu_memory_map[key]
            best_gpu = key

    return best_gpu