# Descanning: From Scanned to the Original Images with a Color Correction Diffusion Model

[Junghun Cha ^{1*}](https://github.com/jhcha08), [Ali Haider]() ^{1*}, [Seoyun Yang](), [Hoeyeong Jin](), [Subin Yang](), [A. F. M. Shahab Uddin](https://scholar.google.com/citations?user=Ckkj9gQAAAAJ&hl=en), [Jaehyoung Kim](https://github.com/crux153), [Soo Ye Kim](https://sites.google.com/view/sooyekim), [Sung-Ho Bae](https://scholar.google.co.kr/citations?user=EULut5oAAAAJ&hl=ko)

Kyung Hee University, Republic of Korea

Jashore University of Science and Technology, Bangladesh

Adobe Research, USA

## Abstract

A significant volume of analog information, i.e., documents and images, have been digitized in the form of scanned copies for storing, sharing, and/or analyzing in the digital world. However, the quality of such contents is severely degraded by various distortions caused by printing, storing, and scanning processes in the physical world. Although restoring high-quality content from scanned copies has become an indispensable task for many products, it has not been systematically explored, and to the best of our knowledge, no public datasets are available. In this paper, we define this problem as Descanning and introduce a new high-quality and large-scale dataset named DESCAN-18K. It contains 18K pairs of original and scanned images collected in the wild containing multiple complex degradations. In order to eliminate such complex degradations, we propose a new image restoration model called DescanDiffusion consisting of a color encoder that corrects the global color degradation and a conditional denoising diffusion probabilistic model (DDPM) that removes local degradations. To further improve the generalization ability of DescanDiffusion, we also design a synthetic data generation scheme by reproducing prominent degradations in scanned images. We demonstrate that our DescanDiffusion outperforms other baselines including commercial restoration products, objectively and subjectively, via comprehensive experiments and analyses.

## DESCAN-18K Example Images

![degradation_final2](https://github.com/jhcha08/Descanning/assets/55647934/1fb77feb-8b8e-4457-b98f-ca5a53e9b79c)

## DescanDiffusion Architecture

![model_final](https://github.com/jhcha08/Descanning/assets/55647934/553407bc-75a4-482d-a800-105cbe7d567e)

## Qualitative Comparisons

![comparison_final2](https://github.com/jhcha08/Descanning/assets/55647934/7cebc99c-1417-479c-a858-2199905ed631)

---

This repository is the official implementation of Descanning: From Scanned to the Original Images with a Color Correction Diffusion Model.

# Updated Soon
