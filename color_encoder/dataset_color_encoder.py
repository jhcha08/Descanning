import random
from os import listdir
from os.path import isfile, join

import natsort
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class ColorEncoderDataset(Dataset):
    def __init__(self, parent_dir):
        self.dir_clean, self.dir_scan = self.image_paths(parent_dir)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(512)])

    def load_img(self, img_path):
        return Image.open(img_path)

    def image_paths(self, parent_dir):
        clean_path = join(parent_dir, 'clean')
        scan_path = join(parent_dir, 'scan')
        
        sorted_files = natsort.natsorted(listdir(clean_path), reverse=True)
        dir_clean = [join(clean_path, f) for f in sorted_files if isfile(join(clean_path, f))]
        dir_scan = [join(scan_path, f) for f in sorted_files if isfile(join(scan_path, f))]

        return dir_clean, dir_scan

    def __getitem__(self, index):
        np.random.seed()
        img_clean = self.load_img(self.dir_clean[index])
        img_scan = self.load_img(self.dir_scan[index])

        if random.random() > 0.5:
            img_clean, img_scan = ImageOps.flip(img_clean), ImageOps.flip(img_scan)
        if random.random() > 0.5:
            img_clean, img_scan = ImageOps.mirror(img_clean), ImageOps.mirror(img_scan)

        img_clean, img_scan = self.transform(img_clean), self.transform(img_scan)
        colors_mean_std = torch.tensor([torch.mean(img_clean[i, :, :]) for i in range(3)] + 
                                        [torch.std(img_clean[i, :, :]) for i in range(3)])
        
        return img_scan, colors_mean_std

    def __len__(self):
        return len(self.dir_clean)