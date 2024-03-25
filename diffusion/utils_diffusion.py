import os
from os import listdir
from os.path import join
from PIL import Image
from tqdm.auto import tqdm
import natsort

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models

from conditional_reverse_function import Unet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
# --------------------------------------------
# Load diffusion model and color encoder for inference
# --------------------------------------------
'''

def load_diffusion_and_encoder(diffusion_weights_path, color_encoder_path):
    model = Unet(dim=64, channels=6, out_dim=3, dim_mults=(1, 2, 4, 8, 8)).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model_checkpoint = torch.load(diffusion_weights_path, map_location=device)
    model.load_state_dict(model_checkpoint['model_state_dict'])

    color_encoder = models.resnet34(pretrained=False)
    color_encoder.fc = nn.Linear(in_features=512, out_features=6, bias=False)
    color_encoder = color_encoder.to(device)
    encoder_checkpoint = torch.load(color_encoder_path, map_location=device)
    color_encoder.load_state_dict(encoder_checkpoint)
    
    return model, color_encoder

'''
# --------------------------------------------
# Util functions to train diffusion model
# --------------------------------------------
'''

def beta_schedule(min_beta=0.0001, max_beta=0.01, steps=2000, schedule='LINEAR'):
    if schedule == 'LINEAR':
        beta = torch.linspace(min_beta, max_beta, steps).to(device)
    return beta

def alpha_t(beta):
    alpha = 1 - beta
    return alpha

def commulative_alpha(alpha):
    alpha_hat = torch.cumprod(alpha,dim=0)
    return alpha_hat

def prepare_alpha_schedules(beta):
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return alpha, alpha_hat.to(device)

def forward_process_step(x_o, alpha_hat, time_step):
    noise = torch.randn_like(x_o)
    xt = (torch.sqrt(alpha_hat[time_step - 1]) * x_o) + (torch.sqrt(1 - alpha_hat[time_step - 1]) * noise)
    return xt, noise


'''
# --------------------------------------------
# Util functions related to image process and performace
# --------------------------------------------
'''

def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(torch.mean((img1 - img2) ** 2)))

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(img):
    return (img + 1) * 0.5

def mean_std_image(image):
    return torch.mean(image, dim=[1, 2]), torch.std(image, dim=[1, 2])

def color_shift(scan_image, mean_clean, std_clean, mean_scan, std_scan):
    normalized_scan = (scan_image - mean_scan[:, None, None]) / std_scan[:, None, None]
    shifted_scan = normalized_scan * std_clean[:, None, None] + mean_clean[:, None, None]
    shifted_scan = (shifted_scan - shifted_scan.min()) / (shifted_scan.max() - shifted_scan.min())
    
    return shifted_scan

'''
# --------------------------------------------
# Sampling and Test codes
# --------------------------------------------
'''

def sampling(denoising_function, img_scan, color_dist, initial_state, alpha, alpha_hat, steps, device):
    img_scan, color_dist = img_scan.to(device), color_dist.to(device)
    x_t = initial_state.to(device)

    for step in tqdm(reversed(range(steps)), desc='Sampling process', total=steps):
        a_t, a_t_hat = alpha[step].to(device), alpha_hat[step].to(device)
        inp = torch.cat((x_t[0] if len(x_t.shape) > 3 else x_t, img_scan), 0)
        noise = torch.randn_like(x_t).to(device) if step > 0 else 0
        correct = ((1 - a_t) / torch.sqrt(1 - a_t_hat)) * denoising_function(inp[None, :], torch.tensor([step + 1], dtype=torch.float32, device=device), color_dist[None, :])
        x_t = x_t - correct + torch.sqrt(1 - a_t) * noise

    return x_t

def test(test_folder, model_function, color_encoder, alpha, alpha_hat, steps):
    newpath = '../test_DescanDiffusion/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    color_encoder.eval()
    model_function.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(512)])
    
    for x in natsort.natsorted(listdir(join(test_folder, 'scan')), reverse=True):
        with torch.no_grad():
            img_scan = transform(Image.open(join(test_folder, 'scan', x))).to(device)
            output_dist = color_encoder(img_scan[None, ...])
            mean_pred, std_pred = output_dist[0][:3], output_dist[0][3:]
            
            mean_scan, std_scan = mean_std_image(img_scan)
            image_scan_shift = color_shift(img_scan, mean_pred, std_pred, mean_scan, std_scan)
            
            image_scan_shift = normalize_to_neg_one_to_one(image_scan_shift)
            initial_state = image_scan_shift
            
            img_descanned = sampling(model_function, image_scan_shift, output_dist[0], initial_state, alpha, alpha_hat, steps, device).clamp_(-1, 1)
            img_descanned = unnormalize_to_zero_to_one(img_descanned)
            
            torchvision.utils.save_image(img_descanned, os.path.join(newpath, x))