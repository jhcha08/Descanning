import os
import sys

import random
import numpy as np
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from dataset_diffusion import DiffusionDataset
from diffusion_trainer import train, weights_init
from conditional_reverse_function import Unet


def main(logging=False):

    if logging:
        time = datetime.now().strftime("%y%m%d_%H%M%S")
        sys.stdout = open("log_DescanDiffusion_{}.txt".format(time), "w")

    train_path = '../dataset/train'
    valid_path = '../dataset/valid'
    color_encoder_path = '../weights_final/color_encoder.h5'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = [0, 1, 2, 3]
    print("Device: ", device)

    seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('Random seed: {}'.format(seed))

    batch_size = 8
    steps = 2000
    sampling_steps = 10
    min_beta = 0.0001
    max_beta = 0.01
    distort_threshold = 0.25
    num_epochs = 5
    save_epoch = 1
    epoch_done = 0
    lr = 0.0001

    diffusion_dataset = DiffusionDataset(train_path, min_beta, max_beta, steps, distort_threshold)
    train_dataloader = DataLoader(diffusion_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print('The number of Data: ', len(diffusion_dataset))

    color_encoder = models.resnet34(pretrained=False)
    color_encoder.fc = nn.Linear(512, 6, bias=False)
    color_encoder.load_state_dict(torch.load(color_encoder_path))
    color_encoder.to(device)

    unet = Unet(dim=64, channels=6, out_dim=3, dim_mults=(1, 2, 4, 8, 8))
    unet.apply(weights_init)
    unet.to(device)
    model = nn.DataParallel(unet, device_ids=gpu_ids)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = train(model, color_encoder, optimizer, criterion, epoch_done, valid_path, 
            train_dataloader, diffusion_dataset, num_epochs, save_epoch, sampling_steps, device)
        
    # Save last model
    weight_path = '../weights_final'
    os.makedirs(weight_path, exist_ok=True)
    torch.save(model.state_dict(), f'{weight_path}/DescanDiffusion.pth')

    if logging:
        sys.stdout.close()

if __name__ == '__main__':
    main()
