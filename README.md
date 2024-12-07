# *LVCD:* Reference-based Lineart Video Colorization with Diffusion Models

### Complete Process from Installation to Usage

#### Detailed Steps

1. **Clone the Project**
   ```bash
   git clone https://github.com/svjack/LVCD && cd LVCD
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   conda create -n lvcd python=3.10.0
   conda activate lvcd
   ```

3. **Install Dependencies**
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name lvcd --display-name "lvcd"
   pip install -r requirements/pt2.txt
   ```

4. **Download Pre-trained Models**
   ```bash
   mkdir ./checkpoints/
   wget https://cdn-lfs-us-1.hf.co/repos/27/21/27217cffe471d846b9341b84b55f3b5be468d886e8df86049bdbb2b13d348afa/6f964edfb7225b4ca74e263f90b4ad8383cdb543fdedb44bed2e7d050e382cbe?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27lvcd.ckpt%3B+filename%3D%22lvcd.ckpt%22%3B&Expires=1733831190&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczMzgzMTE5MH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzI3LzIxLzI3MjE3Y2ZmZTQ3MWQ4NDZiOTM0MWI4NGI1NWYzYjViZTQ2OGQ4ODZlOGRmODYwNDliZGJiMmIxM2QzNDhhZmEvNmY5NjRlZGZiNzIyNWI0Y2E3NGUyNjNmOTBiNGFkODM4M2NkYjU0M2ZkZWRiNDRiZWQyZTdkMDUwZTM4MmNiZT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=OSuaFCgzPDuzBPeU9cYiqc5QuINK1CPEJVOL65BvibchCrtECSeyYyaUqiN5uHv47b3DcvZxc%7ETDz-8XtYmLVUo2CJuwg3ph9s-sUUShXSKSLRRBQ8lTDFsRRv3dpgfEn8z0MRW099sku-XIJd2hdQSnxg9MIdWxXO1a6OijUD%7E8Ffpnqzb4nXSDDjpIs8lDOVq4fG5WFWL28giX%7EA9em6A-IyQ5aTNt-L5bR3kkekSriUxaukD6T9yZvg3iELiMtcLpN%7EStODYgaxn7oeObUOyyWHVg%7ERmsopa2Yy1exfUZNLc6D2HDFIGUYz1UxLFy6GHZi3rtgG9mOzStMbZBig__&Key-Pair-Id=K24J24Z295AEI9
   wget https://cdn-lfs-us-1.hf.co/repos/7e/f0/7ef086cede3588849d02a4ce93c0ab4ab9777d9771b33aaaa53ad7cb3eda786e/3e0994626df395a3831de024f11b2d9d241143bb6f16e2efbacced248aa18ce0?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27svd.safetensors%3B+filename%3D%22svd.safetensors%22%3B&Expires=1733831189&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczMzgzMTE4OX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzdlL2YwLzdlZjA4NmNlZGUzNTg4ODQ5ZDAyYTRjZTkzYzBhYjRhYjk3NzdkOTc3MWIzM2FhYWE1M2FkN2NiM2VkYTc4NmUvM2UwOTk0NjI2ZGYzOTVhMzgzMWRlMDI0ZjExYjJkOWQyNDExNDNiYjZmMTZlMmVmYmFjY2VkMjQ4YWExOGNlMD9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=tpW8ayO7g055lIPscXpW-sddBGHGOMS3kKoOVMBAIbkXj9crwIDA-vnLd%7Esoq0ykkLfg-pThbh24NGvR%7EhsV9-g-o2ciWauGbUEWQNxN7JSMyO2iz56jfqjDTZby8Fex37ExE9jxWNaU7YTD01S8Fb93y5yGOTl5rQstSpFnF8uUzYbWCyg0vDi1IONDiheOgpt%7EZjFBKl1%7E7p%7EXzK6Fe9AUM4zH2GEaIsvCblr7iG20ywjNaiFpZfocx2Mj8TDM%7E3vE8TJc0Mh5-g4D7EEkqEawoZZ36EwOXzKd7KeNvl%7EDkQjT5k21Ros3lbFHt-5Ef3bKYCgF5Y8LZnXLrB50Fw__&Key-Pair-Id=K24J24Z295AEI9
   cp svd.safetensors ./checkpoints
   cp lvcd.ckpt ./checkpoints
   ```

5. **Run Video Generation Task**

```python
import os
import sys
import torch
import numpy as np
import argparse
from PIL import Image
from glob import glob
from utils import load_model
from lineart_extractor.annotator.lineart import LineartDetector
from sample_func import sample_video, decode_video

def run_video_generation(ckpt_path, svd_path, config_path, device, root_dir, output_dir, fps=20, verbose=True):
    """
    Run the video generation task.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        svd_path (str): Path to the SVD model file.
        config_path (str): Path to the configuration file.
        device (str): Device to use for computation (e.g., 'cuda:0').
        root_dir (str): Directory containing input images.
        output_dir (str): Directory to save the output video.
        fps (int): Frames per second for the output video.
        verbose (bool): Whether to print verbose output.
    """
    # Load the model
    model = load_model(device, config_path, svd_path, ckpt_path, use_xformer=True)

    # Initialize the lineart detector
    detector = LineartDetector(device)

    # Load input images and process them
    N = len(glob(f'{root_dir}/*.png'))
    inp = argparse.ArgumentParser()
    inp.resolution = [320, 576]
    inp.imgs = []
    inp.skts = []

    for i in range(N):
        img = load_img(f'{root_dir}/{i}.png', inp.resolution).to(device).unsqueeze(0)
        inp.imgs.append(img)
        np_img = np.array(Image.open(f'{root_dir}/{i}.png').convert('RGB'))
        with torch.no_grad():
            skt = detector(np_img, coarse=False)
        skt = torch.from_numpy(skt).float()
        skt = (skt / 255.0)
        skt = skt[None, None, :, :].repeat(1, 3, 1, 1)
        skt = 1.0 - skt
        inp.skts.append(skt)

    # Set up arguments for video generation
    arg = argparse.ArgumentParser()
    arg.ref_mode = 'prevref'
    arg.num_frames = 19
    arg.num_steps = 25
    arg.overlap = 4
    arg.prev_attn_steps = 25
    arg.scale = [1.0, 1.0]
    arg.seed = 1234
    arg.decoding_t = 10
    arg.decoding_olap = 3
    arg.decoding_first = 1
    arg.fps_id = 6
    arg.motion_bucket_id = 160
    arg.cond_aug = 0.0

    # Generate the video
    sample = sample_video(model, device, inp, arg, verbose=verbose)
    frames = decode_video(model, device, sample, arg)

    # Save the output video
    make_video(output_dir, frames.unsqueeze(0), fps=fps, cols=1, name='output')

# Example usage
ckpt_path = './checkpoints/lvcd.ckpt'
svd_path = './checkpoints/svd.safetensors'
config_path = './configs/lvcd.yaml'
device = 'cuda:0'
root_dir = './inference/test/sample_1'
output_dir = './inference'

run_video_generation(ckpt_path, svd_path, config_path, device, root_dir, output_dir)
```

This script encapsulates the entire process from setting up the environment to running the video generation task. The `run_video_generation` function is designed to be modular and reusable, allowing you to easily adapt it to different projects or configurations.

## ACM Transactions on graphics & SIGGRAPH Asia 2024

[Project page](https://luckyhzt.github.io/lvcd) | [arXiv](https://arxiv.org/abs/2409.12960)

Zhitong Huang $^1$, Mohan Zhang $^2$, [Jing Liao](https://scholars.cityu.edu.hk/en/persons/jing-liao(45757c38-f737-420d-8a7f-73b58d30c1fd).html) $^{1*}$

<font size="1"> $^1$: City University of Hong Kong, Hong Kong SAR, China &nbsp;&nbsp; $^2$: WeChat, Tencent Inc., Shenzhen, China </font> \
<font size="1"> $^*$: Corresponding author </font>

## Abstract:
We propose the first video diffusion framework for reference-based lineart video colorization. Unlike previous works that rely solely on image generative models to colorize lineart frame by frame, our approach leverages a large-scale pretrained video diffusion model to generate colorized animation videos. This approach leads to more temporally consistent results and is better equipped to handle large motions. Firstly, we introduce <em>Sketch-guided ControlNet</em> which provides additional control to finetune an image-to-video diffusion model for controllable video synthesis, enabling the generation of animation videos conditioned on lineart. We then propose <em>Reference Attention</em> to facilitate the transfer of colors from the reference frame to other frames containing fast and expansive motions. Finally, we present a novel scheme for sequential sampling, incorporating the <em>Overlapped Blending Module</em> and <em>Prev-Reference Attention</em>, to extend the video diffusion model beyond its original fixed-length limitation for long video colorization. Both qualitative and quantitative results demonstrate that our method significantly outperforms state-of-the-art techniques in terms of frame and video quality, as well as temporal consistency. Moreover, our method is capable of generating high-quality, long temporal-consistent animation videos with large motions, which is not achievable in previous works.





# Installation

```shell
conda create -n lvcd python=3.10.0
conda activate lvcd
pip3 install -r requirements/pt2.txt
```

# Download pretrained models
1. Download the pretrained [SVD weights](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors) and put it as `./checkpoints/svd.safetensors`
2. Download the finetuned weights for [Sketch-guided ControlNet](https://huggingface.co/luckyhzt/lvcd_pretrained_models/resolve/main/lvcd.ckpt) and put is as `./checkpoints/lvcd.ckpt`

# Inference
All the code for inference is placed under `./inference/`, where the jupyter notebook `sample.ipynb` demonstrates how to sample the videos. Two testing clips are also provided.

# Training
## Dataset preparation
Download the training set from [here](https://huggingface.co/datasets/luckyhzt/Animation_video) including the `.zip`, `.z01` to `.z07`, and `train_clips_hist.json` files.

Unzip the zip files and put the json file under the root directory of the dataset as `.../Animation_video/train_clips_hist.json`.

