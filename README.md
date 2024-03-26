# [AAAI 2024] Descanning: From Scanned to the Original Images with a Color Correction Diffusion Model

[Junghun Cha](https://www.linkedin.com/in/junghun-cha-5a102b1bb/)<sup>1*</sup>, [Ali Haider]()<sup>1*</sup>, [Seoyun Yang](https://kr.linkedin.com/in/seoyun-yang-9b1323218)<sup>1</sup>, [Hoeyeong Jin](https://www.linkedin.com/in/hoeyeong-jin-91987026b/)<sup>1</sup>, [Subin Yang]()<sup>1</sup>, [A. F. M. Shahab Uddin](https://scholar.google.com/citations?user=Ckkj9gQAAAAJ&hl=en)<sup>2</sup>, [Jaehyoung Kim](https://github.com/crux153)<sup>1</sup>, [Soo Ye Kim](https://sites.google.com/view/sooyekim)<sup>3</sup>, [Sung-Ho Bae](https://scholar.google.co.kr/citations?user=EULut5oAAAAJ&hl=ko)<sup>1</sup>

<sup>1</sup> Kyung Hee University, Republic of Korea  
<sup>2</sup> Jashore University of Science and Technology, Bangladesh  
<sup>3</sup> Adobe Research, USA  

This repository is the official PyTorch implementation of "Descanning: From Scanned to the Original Images with a Color Correction Diffusion Model".

[[Paper](https://www.arxiv.org/abs/2402.05350)]  [[Dataset](https://github.com/jhcha08/Descanning/blob/main/dataset/readme.md)]   [[Pre-trained Model](https://drive.google.com/file/d/1neAS5Sh97dlxTFrh9Sn4-kAYvTGVCe0q/view?usp=sharing)]

## Dataset: DESCAN-18K

![degradation_final2](https://github.com/jhcha08/Descanning/assets/55647934/1fb77feb-8b8e-4457-b98f-ca5a53e9b79c)

You can download our dataset DESCAN-18K from these links: [scan1](https://drive.google.com/file/d/1Uanl0NPtVxVOwGb3yzGviopW-j0Gktc6/view?usp=sharing), [scan2](https://drive.google.com/file/d/16DxzIizRdxzrul1T-dgoIzhDn9szFpvK/view?usp=sharing), [clean](https://drive.google.com/file/d/1uB8rFMOjokdYz2ynSPHnxqgqpOAEW707/view?usp=sharing), [validation and test](https://drive.google.com/file/d/12txQIib3ycHcl4f8DscziVtRdN0qZZw1/view?usp=sharing)

After downloading the dataset, please follow detailed instructions [here](https://github.com/jhcha08/Descanning/blob/main/dataset/readme.md).

## Requirements

```
python >= 3.8  
torch >= 1.10.2
torchvision >= 0.11.3  
tqdm >= 4.62.2  
numpy >= 1.22.1  
opencv-python >= 4.5.4.60  
natsort >= 8.1.0  
matplotlib >= 3.4.3  
Pillow >= 9.4.0  
scipy >= 1.7.3  
scikit-image >= 0.16.2  
```

```
pip install -r requirements.txt
```

## Training

To train DescanDiffusion, it is needed to train two modules: Color Encoder and Conditional DDPM.

### Color Encoder (Global Color Correction)

1. Configure settings in ```color_encoder/train_color_encoder.py```. (e.g. dataset path, batch size, epochs).
 - If you want to log the training process, set ```logging=True```.  
2. Execute the below code to train the color encoder.
   ```
   python3 color_encoder/train_color_encoder.py
   ```
3. The last saved model will become ```color_encoder.h5```. It will used to train the conditional DDPM (below part).

### Conditional DDPM (Local Generative Refinement)

1. Configure settings in ```diffusion/train_diffusion.py```. (e.g. dataset path, pre-trained color encoder path, batch size, epochs, steps).  
 - If you want to log the training process, set ```logging=True```.  
 - If you want to adjust synthetic data generation probability, adjust ```distort_threshold```. (It is 0.25 in the paper).
2. Execute the below code to train the conditional DDPM.
   ```
   python3 diffusion/train_diffusion.py
   ```
3. The last saved model will become ```DescanDiffusion.pth```. It will used to infer scanned images of the testing set.

## Testing

---

## Abstract

A significant volume of analog information, i.e., documents and images, have been digitized in the form of scanned copies for storing, sharing, and/or analyzing in the digital world. However, the quality of such contents is severely degraded by various distortions caused by printing, storing, and scanning processes in the physical world. Although restoring high-quality content from scanned copies has become an indispensable task for many products, it has not been systematically explored, and to the best of our knowledge, no public datasets are available. In this paper, we define this problem as **Descanning** and introduce a new high-quality and large-scale dataset named **DESCAN-18K**. It contains 18K pairs of original and scanned images collected in the wild containing multiple complex degradations. In order to eliminate such complex degradations, we propose a new image restoration model called **DescanDiffusion** consisting of a color encoder that corrects the global color degradation and a conditional denoising diffusion probabilistic model (DDPM) that removes local degradations. To further improve the generalization ability of DescanDiffusion, we also design a synthetic data generation scheme by reproducing prominent degradations in scanned images. We demonstrate that our DescanDiffusion outperforms other baselines including commercial restoration products, objectively and subjectively, via comprehensive experiments and analyses.

## DescanDiffusion Architecture

![model_final](https://github.com/jhcha08/Descanning/assets/55647934/553407bc-75a4-482d-a800-105cbe7d567e)

## Qualitative Comparisons

![comparison_final2](https://github.com/jhcha08/Descanning/assets/55647934/7cebc99c-1417-479c-a858-2199905ed631)

## News

✨ [2024-03-25] Our codes and pre-trained models are released.  
📥 [2024-03-25] Our dataset DESCAN-18K is released.  
📃 [2024-02-08] Our paper is uploaded on arXiv.  
🎉 [2023-12-09] Our paper is accepted by AAAI 2024.  

## Citation

If this repository is useful to your research, please consider citing our works! 😊

```
@article{cha2024descanning,
        title={Descanning: From Scanned to the Original Images with a Color Correction Diffusion Model},
        author={Cha, Junghun and Haider, Ali and Yang, Seoyun and Jin, Hoeyeong and Yang, Subin
                and Uddin, AFM and Kim, Jaehyoung and Kim, Soo Ye and Bae, Sung-Ho},
        journal={arXiv preprint arXiv:2402.05350},
        year={2024}
}
```
