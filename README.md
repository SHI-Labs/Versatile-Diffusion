# Versatile Diffusion

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repo hosts the official implementary of:

[Xingqian Xu](https://ifp-uiuc.github.io/), Atlas Wang, Eric Zhang, Kai Wang, and [Humphrey Shi](https://www.humphreyshi.com/home), **Versatile Diffusion: Text, Images and Variations All in One Diffusion Model**, [Paper arXiv Link coming soon].

## News

- [2022.11.14]: Part of our evaluation codes and models are released!
- [2022.11.12]: Repo initiated

## Introduction

<p align="center">
  <img src="assets/figures/teaser.png" width="99%">
</p>


**Versatile Diffusion (VD)** is a four-flow diffusion model that parallely handles text-to-image, image-variation, image-to-text, and text-variation. From which we extended to a generalized multi-flow multimodal framework that can further be expanded into other modalities and other tasks, such as image-to-audio, audio-to-image.

## Network and Framework

One single flow of VD contains a VAE, a diffusor and a context encoder, and thus handles one tasks (e.g. text-to-image) under one data type (e.g. image) and one context type (e.g. text). And the multi-flow structure of VD shows in the following diagram:

<p align="center">
  <img src="assets/figures/VD_framework.png" width="99%">
</p>

According to VD, we further proposed a generalized multi-flow multimodal framework with VAEs, context encoders, and diffusors that contains three types of layers (i.e. global, data, context layers). To involve a new multimodal task in this framework, we bring out the following requirements:

* The design of core diffusor should contain shared global layers, swappable data and context layers that will be correspondingly activated based on data and context types.
* The choice of VAEs should smoothly map data onto highly interpretable latent spaces.
* The choice of context encoders should jointly minimize the cross-modal statistical distance on all supported content types.


## Performance

<p align="center">
  <img src="assets/figures/qcompare1.png" width="99%">
  <img src="assets/figures/qcompare2.png" width="99%">
  <img src="assets/figures/qcompare3.png" width="99%">
</p>

## Data

We use Laion2B-en with customized data filters as our main dataset. Since Laion2B is very large and typical trainings are less than one epoch, so usually we don't need to download the full dataset for training. Same story for VDs.

Directory of Laion2B for our code:

```
├── data
│   └── laion2b
│       └── data
│           └── 00000.tar
│           └── 00000.parquet
│           └── 00000_stats.jsom_
│           └── 00001.tar
│           └── ...
```

These compressed data is generate with img2dataset API [official github link](https://github.com/rom1504/img2dataset).

## Setup

```
conda create -n versatile-diffusion python=3.8
conda activate versatile-diffusion
conda install pytorch==1.12.1 torchvision=0.13.1 -c pytorch
pip install -r requirement.txt
```

## Pretrained models

All useful pretrained model can be downloaded from this [link](https://drive.google.com/drive/folders/1SloRnOO9UnonfvubPWfw0uFpLco_2JvH?usp=sharing). The pretrained folder should include the following files:

```
├── pretrained
│   └── kl-f8.pth
│   └── optimus-vae.pth
│   └── sd-v1-4.pth
│   └── sd-variation-ema.pth
│   └── vd-dc.pth
│   └── vd-official.pth
```

## Evaluation

Here are the one-line shell commends to evaluation SD baselines with mutliple GPUs.

```
python main.py --config sd_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
python main.py --config sd_variation_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
```

Here are the one-line shell commends to evaluation VD models on multiple GPUs.

```
python main.py --config vd_dc_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
python main.py --config vd_official_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
```

All corresponding evaluation configs can be found in ```./configs/experiment```. There are many useful information in the config. You can easy customized it and run your own batched evaluations.

For the commends above, you also need to:
* Create ```./pretrained``` and move all downloaded pretrained models in it.
* Create ```./log/sd_nodataset/99999_eval``` for baseline evaluations on SD
* Create ```./log/vd_nodataset/99999_eval``` for evaluations on VD

## Training

Coming soon

## Citation

Coming soon

## Acknowledgement

Part of the codes reorganizes/reimplements code from the following repositories: [LDM official Github](https://github.com/CompVis/latent-diffusion), which also oriented from [DDPM official Github](https://github.com/lucidrains/denoising-diffusion-pytorch).
