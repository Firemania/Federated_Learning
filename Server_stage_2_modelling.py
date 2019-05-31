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
from lib.tasks import get_data_transforms, load_data, train_model, model_take_lower_layers, feature_eval_prep
from lib.Inception_V3_trimmed import inception_v3_trimmed
from lib.tasks import feature_model_data_transforms, refolder

### This script is to train model on the extracted feature data from clients. Feature data
### from all clients are integrated together as one data set before running this code.

## Loading the dataloaders -- Make sure that the data is saved in following way
"""
data/
  - train/
      - class_1 folder/
          - img1.png
          - img2.png
      - class_2 folder/
      .....
      - class_n folder/
  - val/
      - class_1 folder/
      - class_2 folder/
      ......
      - class_n folder/
"""







parser = argparse.ArgumentParser(description='PyTorch inception Training')
parser.add_argument('--img_dir', default='./features/', type=str, help='')
parser.add_argument('--reorder', default=True, type=bool, help='whether to split data into train val test')
parser.add_argument('--output_dir', default='./checkpoint/', type=str, help='')
parser.add_argument('--restore', default=0, type=int, help='whether restore the stored model or not')
parser.add_argument('--final_output', default=2, type=int, help='the number of the output of the final layer')
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--epochs', default=30, type=int)

args = parser.parse_args()
if args.reorder==True:
    """""""""""
    set up folder
    """""""""""
    refolder(args.img_dir, 'new_data_folder', train_fraction=0.8, val_fraction=0.2, test_fraction=0.0, 
                remove_original=False)
    data_dir = 'new_data_folder'
else:
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

##### Important: here the data_trainsforms is special
data_transforms = feature_model_data_transforms(mode="train")

dataloaders, dataset_sizes, class_names, class_2_idx = load_data(data_dir, 
                                                                 data_transforms, 
                                                                 mode = 'train', 
                                                                 batch_size=batch_size, 
                                                                 num_workers=None)

# Get truncated model that starts from a higher level layer. The model currently 
# used is Inception V3 with first 3 layers cut off. The model starts from the 
# pooling layer before Layer 4

if(restore):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    model_conv=torch.load('checkpoint/base_model_1.t7')
    model_conv=model_conv.module
else:
    model_conv = inception_v3_trimmed(pretrained=True)


freeze_layers=1
if freeze_layers:
  for i, param in model_conv.named_parameters():
    param.requires_grad = False
# Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, n_class)

# Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
ct = []
for name, child in model_conv.named_children():
    # Mixed_7c and higher are the layers to be retrained
    if "Mixed_7c" in ct:
        for params in child.parameters():
            params.requires_grad = True
    ct.append(name)




if use_gpu:
    model_conv.cuda()
    print("[Using all the available GPUs]")
    if use_parallel:
        model_conv = nn.DataParallel(model_conv, device_ids=[0, 1, 2, 3])
        model_conv.cuda()
    print("[Using CrossEntropyLoss...]")
    criterion = nn.CrossEntropyLoss().cuda()
else:
    print("[Using CrossEntropyLoss...]")
    criterion = nn.CrossEntropyLoss()

print("[Using small learning rate with momentum...]")
optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr=0.005, momentum=0.9)

print("[Creating Learning rate scheduler...]")
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print("[Training the model begins ....]")
train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer, 
            scheduler, use_gpu, epochs, output_dir, mixup = False, alpha = 0.1)

# Then send new model to the clients if necessary.

