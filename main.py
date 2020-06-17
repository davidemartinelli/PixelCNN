import torch
from torch.utils.data import DataLoader

import torchvision

from model import PixelCNN
from args import args
from utils import BinaryMNIST

import matplotlib.pyplot as plt

import time
import os

IMAGES_FOLDER = 'generated_images'
BIAS = False 

train_loader = DataLoader(BinaryMNIST(), batch_size=args.b, shuffle=True, num_workers=1, pin_memory=True)
test_loader = DataLoader(BinaryMNIST(train=False), batch_size=args.b, shuffle=False, num_workers=1, pin_memory=True)

model = PixelCNN(args.ch, args.hl, args.k, BIAS)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

start = time.time()

for i in range(args.e):
    print(f'Epoch {i + 1}...')
    
    #training
    for images, labels in train_loader:
        images = images.to(model.device)
        out = model(images) #(batch_size, 1, 28, 28)

        neg_log_likelihood = -(images * torch.log(out + 1e-7) + (1 - images) * torch.log(1 - out + 1e-7)).mean()
        
        optimizer.zero_grad()
        neg_log_likelihood.backward()
        optimizer.step()

    #validation
    for images, labels in test_loader:
        model.eval()

        images = images.to(model.device)
        out = model(images) #(batch_size, 1, 28, 28)

        neg_log_likelihood = -(images * torch.log(out + 1e-7) + (1 - images) * torch.log(1 - out + 1e-7)).mean()

        model.train()
        
    print('Validation negative log likelihood:', neg_log_likelihood.item())

    end = time.time()
    print(f'Done in {round(end-start)} seconds')
    start = end

print('Generating images from the model... ', end='')
samples = model.sample(n_samples=144)

if not os.path.exists(IMAGES_FOLDER):
            os.mkdir(IMAGES_FOLDER)

torchvision.utils.save_image(samples, os.path.join(IMAGES_FOLDER, f'samples_k_{args.k}_c_{args.ch}_d_{args.hl}_e_{args.e}.png'), nrow=12, padding=0)

print('Done!')