import numpy as np
from PIL import Image
import torch
import torchvision
from torch import nn


def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(torch.mean((img1 - img2) ** 2)))

def initialize_model(device):
    model = torchvision.models.resnet34(pretrained=False, progress=True)
    model.fc = nn.Linear(in_features=512, out_features=6, bias=False)
    print('Device: ', device)

    return model.to(device)

def calculate_mean_std(image):
    r_channel, g_channel, b_channel = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    mean = np.mean(r_channel), np.mean(g_channel), np.mean(b_channel)
    std = np.std(r_channel), np.std(g_channel), np.std(b_channel)

    return mean, std

def apply_color_shift(image, clean_mean, clean_std, scan_mean, scan_std):
    normalized_channels = [(image[:, :, i] - scan_mean[i]) / scan_std[i] for i in range(3)]
    shifted_channels = [normalized_channels[i] * clean_std[i] + clean_mean[i] for i in range(3)]
    shifted_image = np.clip(np.stack(shifted_channels, axis=-1), 0, 1)

    return shifted_image

def load_and_preprocess_image(image_path, resize_dims=(512, 512)):
    image = Image.open(image_path)
    image = image.resize(resize_dims)
    image = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
    mean, std = calculate_mean_std(image)

    return image, mean, std