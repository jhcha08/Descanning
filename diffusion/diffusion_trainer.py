import os
import os.path

import numpy as np
import natsort
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.multiprocessing as multiprocessing
multiprocessing.set_sharing_strategy('file_system')
import torchvision
from torchvision import transforms

from utils_diffusion import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)

def train(model, color_encoder, optimizer, criterion, epoch_done, test_folder,
          train_set, diffusion_dataset, num_epochs, save_epoch, sampling_steps, device):
    model.to(device)
    loss_train = []
    epochs_record = []

    for epoch in range(epoch_done + 1, num_epochs + 1):
        print(f'{"-" * 10}\nEpoch {epoch}/{num_epochs}')
        model.train()
        running_loss = 0.0

        for i, (x_t_scan, _, noise, t, color_dist) in enumerate(train_set, start=1):
            x_t_scan, color_dist, noise, t = x_t_scan.to(device), color_dist.to(device), noise.to(device), t.to(device)

            optimizer.zero_grad()
            out = model(x_t_scan, t, color_dist)
            loss = criterion(out, noise)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_t_scan.size(0)

        epoch_loss = running_loss / len(train_set.dataset)
        epochs_record.append(epoch)
        loss_train.append(epoch_loss)

        print(f'\nEPOCH {epoch} - Epoch Loss: {epoch_loss:.4f} - Lr rate: {optimizer.param_groups[0]["lr"]}')

        if epoch % save_epoch == 0:
            model.eval()
            with torch.no_grad():
                validate(test_folder, model, color_encoder, diffusion_dataset, sampling_steps, epoch - 1, device)
                
            weight_path = './weights_diffusion/'
            os.makedirs(weight_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(weight_path, f'diffusion_{epoch}.pth'))

    plt.plot(epochs_record, loss_train, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('Training_DescanDiffusion.png')

    return model

def validate(test_folder, model_function, color_encoder, custom_set, steps, epoch, device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(512)])
    newpath = os.path.join('./validation_diffusion', str(epoch+1))
    os.makedirs(newpath, exist_ok=True)

    total_PSNR = []
    print(f'\n[ Validation at epoch {epoch + 1} ]')

    for x in natsort.natsorted(os.listdir(os.path.join(test_folder, 'scan')), reverse=True):
        img_scan = transform(Image.open(os.path.join(test_folder, 'scan', x))).to(device)
        img_clean = transform(Image.open(os.path.join(test_folder, 'clean', x))).to(device)
        
        # output_dist = r_mean, g_mean, b_mean,r_std, g_std, b_std
        output_dist = color_encoder(img_scan.unsqueeze(0)).squeeze()
        mean_pred, std_pred = output_dist[:3], output_dist[3:]

        mean_scan, std_scan = mean_std_image(img_scan)
        img_scan_shift = color_shift(img_scan, mean_pred, std_pred, mean_scan, std_scan)
        img_scan_shift = normalize_to_neg_one_to_one(img_scan_shift)

        color_dist = torch.tensor([*mean_pred, *std_pred], device=device)
        alpha, alpha_hat = custom_set.alpha.to(device), custom_set.alpha_hat.to(device)
        initial_state = img_scan_shift

        descanned_img = sampling(model_function, img_scan_shift, color_dist, initial_state, alpha, alpha_hat, steps, device).clamp_(-1, 1)
        descanned_img = unnormalize_to_zero_to_one(descanned_img)

        current_psnr = psnr(descanned_img[0], img_clean)
        print(f'Name: {x}, PSNR: {current_psnr.item()}')
        total_PSNR.append(current_psnr.item())

        torchvision.utils.save_image(descanned_img, os.path.join(newpath, f'{x[:-4]}_{epoch+1}.png'))

    print(f'\nAverage Validation PSNR: {np.mean(total_PSNR)}\n')