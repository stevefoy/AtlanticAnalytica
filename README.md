# PlantCLEF 2024

Submission code for Atlantic. Please cite if you find it useful!
<img src="./sample_images/CCC-PPPP-BB-22222222.jpg" name="RGB">

## Overview

This repository contains the code for the PlantCLEF 2024 competition submission by ID Atlantic. The task will be evaluated as a multi-label classification task that aims to predict all the plant species present in high-resolution plot images. 

The main challenge of this task is the shift between the test data (high-resolution multi-label images of vegetation plots) and the training data (single-label images of individual plants).

## Task Description

The objective is to achieve accurate multi-species plant identification in vegetation plot images. This involves:
- Analyzing high-resolution plot images.
- Predicting multiple plant species present in each image.

## Resources

- [ImageCLEF PlantCLEF 2024 Task Overview](https://www.imageclef.org/node/315)
- [Hugging Face PlantCLEF 2024 Space](https://huggingface.co/spaces/BVRA/PlantCLEF2024)


Submisson for ID Atlantic
The task will be evaluated as a multi-label classification task that aims to predict all the plant species on the high-resolution plot images. The main difficulty of the task lies in the shift between the test data (high-resolution multi-label images of vegetation plots) and the training data (single-label images of individual plants).

multi-species plant identification in vegetation plot images

https://www.imageclef.org/node/315
https://huggingface.co/spaces/BVRA/PlantCLEF2024



# Turfgrass Divot analysis with sequoia camera
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)



Code for Plant Clef2024,  implementation for our [paper]()).  If you find this code useful in your research, please consider citing

<!-- TOC -->

- [Divot in PyTorch]
  - [Requirements](#requirements)
  - [Code structure](#code-structure)
  - [Acknowledgement](#acknowledgement)
  - [Citation]

<!-- /TOC -->



## Requirements
Conda was used, config as follows:

```bash
pip install -r requirements.txt
```

or for a local installation

```bash
pip install --user -r requirements.txt
```

### manual install 
conda create enviroment
conda install -c conda-forge pytorch-gpu
conda install torchvision


## Code structure
The code structure overview

  ```
  code-base/
  │
  ├── ./script/sliding_window_inference_pipeline.py - generate database files
  ├── ./script/sliding_window_to_plantclef_csv.py - Data process  
  │
  │
  ├── models/ - contains Dino V2 models from Team CLEF 2024
  │
  │
  ├── database/
  │   ├── test/ - data producted for Image testing 
  │   └── ResultsALL_XXXXW_YYYYS_ZZZZB.csv- all the database X Window size, Y Slide, Z Border pixels
  │
  ```
### ToDo List
- [x] Baseline created
- [x] Conda env setup with repo
- [x] Test Pytorch Vision Dino V2 
- [x] Setup 


## Acknowledgement
- [Dino reference Tutorial](https://github.com/facebookresearch/dino)
- [Facebook SAM code ](https://github.com/facebookresearch/segment-anything)
- [PlantCLEF 2024 pretrained models](https://zenodo.org/records/10848263?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk5MmJkMGMyLWM3NDEtNDZiYi04ODIzLWQ3ODE1M2JjOGIyMiIsImRhdGEiOnt9LCJyYW5kb20iOiIwYTQwOGJjY2Q0NmFiYzZjMGRlMWNkNzZmMzk3NjMxZSJ9.MKZoVFGL6jGoI3TUmnFUzTy5nrPzgf12C1RBip--ECABBsipgctxSp5U4HfRsk9rL4FfJM2Q4_vKAYb1z1Z-Zg)
- [TSNE ]


## Citation

    @inproceedings{PlantCLEF2024,
        author = {Stephen Foy and Simon McLoughlin},
        title = {Slide and Check with Dino V2 PLANT CLEF 2024},
        booktitle = {PlantCLEF 2024},
        year = {2024}
    }



