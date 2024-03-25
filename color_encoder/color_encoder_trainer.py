import os

import natsort
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

from utils_color_encoder import *


def train(model, dataloaders, test_folder, criterion, optimizer, device, num_epochs, save_epoch):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for _, (scan_img, clean_statistics) in enumerate(dataloaders, start=1):
            scan_img, clean_statistics = scan_img.to(device), clean_statistics.to(device)
            optimizer.zero_grad()
            outputs = model(scan_img)
            loss = criterion(outputs, clean_statistics)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * scan_img.size(0)

        epoch_loss = running_loss / len(dataloaders.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.5f} | Learning Rate: {optimizer.param_groups[0]["lr"]:.3e}')

        if (epoch + 1) % save_epoch == 0:
            model.eval()
            with torch.no_grad():
                validate(test_folder, model, device, epoch)

            weight_path = './weights_color_encoder/'
            os.makedirs(weight_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(weight_path, f'color_encoder_{epoch+1}.pth'))

    return model

def validate(test_folder, model_function, device, epoch):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(512)])
    newpath = os.path.join('./validation_color_encoder', str(epoch+1))
    os.makedirs(newpath, exist_ok=True)
    
    total_PSNR = []    
    print(f'\n[ Validation at epoch {epoch+1} ]')

    for x in natsort.natsorted(os.listdir(os.path.join(test_folder, 'scan')), reverse=True):
        img_scan = transform(Image.open(os.path.join(test_folder, 'scan', x))).to(device)

        output = model_function(img_scan[None, ...].to(device))
        [r_mean, g_mean, b_mean, r_std, g_std, b_std] = output[0].cpu().numpy()

        mean_pred = r_mean, g_mean, b_mean
        std_pred = r_std, g_std, b_std

        image_scan_path = os.path.join(test_folder, 'scan', x)
        image_scan, mean_scan, std_scan = load_and_preprocess_image(image_scan_path)

        image_clean_path = os.path.join(test_folder, 'clean', x)
        _, mean_clean, std_clean = load_and_preprocess_image(image_clean_path)

        pred_shift = apply_color_shift(image_scan.copy(), mean_pred, std_pred, mean_scan, std_scan)
        clean_shift = apply_color_shift(image_scan.copy(), mean_clean, std_clean, mean_scan, std_scan)
 
        pred_shift, clean_shift = transform(pred_shift), transform(clean_shift)
        
        psnr_value = psnr(pred_shift, clean_shift).item()
        print(f'Name: {x}  PSNR {psnr_value}')
        total_PSNR.append(psnr_value)

        torchvision.utils.save_image(pred_shift, f'{newpath}/{x[:-4]}_{epoch+1}.png')

    avg_psnr = np.mean(np.array(total_PSNR))
    print(f'\nAverage Validation PSNR: {avg_psnr}\n')