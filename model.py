'''
Basic implementation of PixelCNN for binary vectors.
Some methods are specific to the MNIST dataset. In particular,
the model assumes that the images have size (1, 1, 28, 28).
'''

import torch
import torch.nn as nn

from utils import BinaryMNIST, get_gpu_memory_map

class AutoregressiveConv2d(nn.Conv2d):
    '''
    This class implements a standard nn.Conv2d layer
    with the addition of a binary mask.
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=False, is_first=False):
        assert kernel_size % 2 != 0, 'Kernel size must be odd' #for simplicity
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2,  bias=bias)

        self.initialize_mask(kernel_size, is_first)

    def initialize_mask(self, kernel_size, is_first):
        self.register_buffer('mask', torch.ones(kernel_size, kernel_size))

        self.mask[kernel_size // 2, kernel_size // 2:] = 0
        self.mask[kernel_size // 2 + 1:] = 0

        if not is_first: #type B mask
            self.mask[kernel_size // 2, kernel_size // 2] = 1

        self.weight.data *= self.mask #just to make sure weights are correct at initialization. can be removed though

    def forward(self, x):
        self.weight.data *= self.mask
        
        return self._conv_forward(x, self.weight)

class PixelCNN(nn.Module):
    '''
    This class provides a basic implementation of PixelCNN
    as described in https://arxiv.org/pdf/1601.06759.pdf.
    '''

    def __init__(self, n_channels, n_layers, kernel_size, bias=False):
        assert n_layers > 0, 'The net must have at least one layer!'
        super().__init__()

        self.device = torch.device('cuda:' + str(get_gpu_memory_map()) if torch.cuda.is_available() else 'cpu')

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.initialize_net(n_channels, kernel_size, n_layers, bias)

        self.to(self.device)

    def initialize_net(self, n_channels, kernel_size, n_layers, bias):
        self.layers = nn.ModuleList()

        self.layers.append(AutoregressiveConv2d(1, n_channels, kernel_size, bias=bias, is_first=True))
        self.layers.append(nn.BatchNorm2d(n_channels))
        
        for i in range(n_layers - 1):
            self.layers.append(AutoregressiveConv2d(n_channels, n_channels, kernel_size, bias=bias))
            self.layers.append(nn.BatchNorm2d(n_channels))
        
        self.layers.append(AutoregressiveConv2d(n_channels, 1, kernel_size, bias=True))
   
    def forward(self, x):
        for i,layer in enumerate(self.layers):
            if i != 0 and isinstance(layer, nn.BatchNorm2d):
                x = layer(x)
                x = self.activation(x)

            if isinstance(layer, AutoregressiveConv2d):
                x = layer(x)
        
        return self.sigmoid(x)

    def sample(self, n_samples=144):
        '''
        This function generates images by sampling autoregressively from the model.
        '''

        with torch.no_grad():
            self.eval() #for batch-norm

            images = torch.zeros(n_samples, 1, 28, 28, device=self.device)

            for i in range(784):
                out = self(images).view(n_samples, 1, -1)
                images[:, :, i // 28, i % 28] = torch.bernoulli(out[:, :, i])

            self.train()

            return images
