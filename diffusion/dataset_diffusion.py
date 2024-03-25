import glob
import os
import os.path
import random

import numpy as np
import natsort
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from utils_diffusion import *
from synthesizing_degradations import transform_image, uint2single


class DiffusionDataset(Dataset):
    def __init__(self, parent_dir, min_beta, max_beta, steps, degrading_threshold, schedule='LINEAR'):
        self.clean_path = os.path.join(parent_dir, 'clean')
        self.dir_clean, self.dir_scan = self.image_paths(parent_dir)
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.steps = steps
        self.schedule = schedule
        self.degrading_threshold = degrading_threshold
        self.beta = self.beta_schedule(self.min_beta, self.max_beta, self.steps, self.schedule)
        self.alpha = self.alpha_t(self.beta)
        self.alpha_hat = self.commulative_alpha(self.alpha)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(512)])

    def random_file(self, folder_path):
        files = glob.glob(os.path.join(folder_path, '**', '*'), recursive=True)
        return random.choice(files)

    def load_img(self, img_path):
        return Image.open(img_path)

    def image_paths(self, parent_dir):
        clean_path = os.path.join(parent_dir, 'clean', '*')
        clean_files = [x for x in natsort.natsorted(glob.glob(clean_path), reverse=True) if os.path.isfile(x)]
        scan_files = [os.path.join(parent_dir, 'scan', os.path.basename(x)) for x in clean_files if os.path.isfile(os.path.join(parent_dir, 'scan', os.path.basename(x)))]
        
        return clean_files, scan_files
    
    def forward_process_step(self, x_o, alpha_hat, time_step):
        noise = torch.randn_like(x_o)
        xt = (torch.sqrt(alpha_hat[time_step - 1]) * x_o) + (torch.sqrt(1 - alpha_hat[time_step - 1]) * noise)
        return xt, noise

    def beta_schedule(self, min_beta, max_beta, steps, schedule='LINEAR'):
        if schedule =='LINEAR':
            beta = torch.linspace(min_beta, max_beta, steps)
        return beta

    def alpha_t(self, beta):
        alpha = 1 - beta
        return alpha

    def commulative_alpha(self, alpha):
        alpha_hat = torch.cumprod(alpha, dim=0)
        return alpha_hat
   
    def __len__(self):
        return len(self.dir_scan)
    
    def __getitem__(self, index):
        np.random.seed()
        img_clean = self.load_img(self.dir_clean[index])
        img_scan = self.load_img(self.dir_scan[index])
        blending_target = self.random_file(self.clean_path)
        img_blend = self.load_img(blending_target)

        if random.random() > 0.5:
            img_clean, img_scan = ImageOps.flip(img_clean), ImageOps.flip(img_scan)
        if random.random() > 0.5:
            img_clean, img_scan = ImageOps.mirror(img_clean), ImageOps.mirror(img_scan)

        img_clean, img_scan, img_blend = map(lambda img: np.array(img.convert('RGB'), dtype=np.uint8), [img_clean, img_scan, img_blend])

        # Synthesized vs. Original Scanned
        if random.uniform(0, 1) < self.degrading_threshold:
            img_clean, img_scan = transform_image(img_clean, img_blend)

        img_clean = uint2single(img_clean)
        img_scan = uint2single(img_scan)

        img_clean = self.transform(np.array(img_clean))
        img_scan = self.transform(np.array(img_scan))

        mean_clean, std_clean = mean_std_image(img_clean)
        mean_scan, std_scan = mean_std_image(img_scan)

        img_scan_shift = color_shift(img_scan, mean_clean, std_clean, mean_scan, std_scan)
        
        r, g, b = img_scan_shift[0,:,:], img_scan_shift[1,:,:], img_scan_shift[2,:,:]
        img_clean, img_scan_shift = normalize_to_neg_one_to_one(img_clean), normalize_to_neg_one_to_one(img_scan_shift)

        r_mean, g_mean, b_mean = torch.mean(r), torch.mean(g), torch.mean(b)
        r_std, g_std, b_std = torch.std(r), torch.std(g), torch.std(b)
        
        color_dist = torch.tensor([r_mean, g_mean, b_mean, r_std, g_std, b_std])
        
        t = random.randint(1,self.steps)

        x_t, noise = self.forward_process_step(img_clean, self.alpha_hat, t)
        x_t_scan = torch.cat((x_t, img_scan_shift),0)

        return x_t_scan, img_scan_shift, noise, t, color_dist