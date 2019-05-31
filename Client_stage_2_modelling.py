from torch.autograd import Variable
from tqdm import tqdm
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
import torchvision
import copy
from lib.tasks import (get_data_transforms, load_data, train_model, 
                       eval_model, model_take_lower_layers, feature_eval_prep)

### This script is to complete the whole process of extracting features from a set of labeled image data.
## Loading the dataloaders -- Make sure that the data is saved in following way
"""

  - data/
      - class_1 folder/
          - img1.png
          - img2.png
      - class_2 folder/
      .....
      - class_n folder/
  
"""
parser = argparse.ArgumentParser(description='PyTorch inception Training')
parser.add_argument('--img_dir', default='./train/', type=str, help='')
parser.add_argument('--output_dir', default='./features/', type=str, help='')
parser.add_argument('--restore', default=1, type=int, help='whether restore the stored model or not')
parser.add_argument('--final_output', default=2, type=int, help='the number of the output of the final layer')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--epochs', default=30, type=int)

args = parser.parse_args()
data_dir = args.img_dir
restore=args.restore
n_class=args.final_output
batch_size=args.batch_size
epochs=args.epochs
output_dir=args.output_dir

input_shape = 299
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
scale = 299
input_shape = 299 
use_parallel = True
use_gpu = True

print('lets start')
data_transforms = get_data_transforms(mode="eval", model_origin="Inception V3", 
                          mean=None, std=None, scale=None, input_shape=299)
print('data_transform ends')
print(data_transforms)
dataloaders, dataset_sizes, class_names, class_2_idx = load_data(data_dir, data_transforms, mode = 'eval', batch_size=batch_size, num_workers=1)
print('load finished')
# Get model

if(restore):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    model_conv=torch.load('checkpoint/ckpt_led.t7')
    model_conv=model_conv.module
else:
    model_conv = torchvision.models.inception_v3(pretrained=True)

# Take model with only first ? layers
model_conv = model_take_lower_layers(model_conv, layer_number=3)

if use_parallel and use_gpu:
    print("[Using all the available GPUs]")
    model_conv.cuda()
    model_conv = nn.DataParallel(model_conv, device_ids=[0, 1, 2, 3])
    model_conv.cuda()
    

# Set up feature extraction classes and folders
idx_2_class=feature_eval_prep(class_2_idx, feat_path=output_dir)
print(model_conv)
eval_model(model_conv, dataloaders, dataset_sizes,  use_gpu, batch_size, 
           idx_2_class, feat_path=output_dir)

# Then send feature data to the server

