from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

from torchvision import datasets, models, transforms
from torch.autograd import Variable
from PIL import Image

import os
import ntpath
import sys
import glob
import random
from tqdm import tqdm
import time
import argparse
import torchvision
import copy
import pdb
import shutil
from shutil import copyfile



# Load pretrained base model

def load_model(model_path="base_model_1.pt"):
    # torch.load() needs pickle.py location. We will add code in future in the case that default does not work.
    try:
        model=torch.load(model_path)
    except Exception as e:
        print("Failed to load model")
        print(e)
    return model


# Save model

def save_model(model, model_path="new_model.pt"):
    # torch.save() needs pickle.py location. We will add code in future in the case that default does not work.
    try:
        torch.save(model, model_path)
    except Exception as e: 
        print("Failed to save model.")
        print(e)


# Use first few layers as a model

def model_take_lower_layers(model, layer_number=3):

    try:
        feat_model=nn.Sequential(*list(model.children())[:layer_number])
    except Exception as e:
        print("Problem separating the model")
        print(e)
        feat_model=None
        
    return feat_model


# data_transforms for different types of tasks

# inception v3 preprocessing as default
def get_data_transforms(mode="train", model_origin="Inception V3", mean=None, std=None, scale=None, input_shape=299):
    # mode: "train", "test", or "eval".
    # model_origin: use given or custom ones.
    if model_origin== "Inception V3":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        scale = 299
    # Use other models, then add 
    #elif model_origin== "??":
    #    mean=....
    
    
    elif model_origin=="custom":
        # Your own parameters passed by mean, std, scale arguments
        pass
    
    if mode =="eval":  # eval is for getting simply feed forward results
        print('creating data_transform')
        data_transforms = {'eval':transforms.Compose([
                    transforms.Resize((scale,scale)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])}
        print('data_transform finished')

    elif mode == "train":  # train is for model training
        data_transforms = {'train': transforms.Compose([
                    transforms.Resize((scale,scale)),
                    transforms.RandomCrop(input_shape),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=45),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)]),
                    'val': transforms.Compose([
                    transforms.Resize((scale, scale)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])}
    elif mode == "test":  # test is for model testing
        data_transforms = {'test': transforms.Compose([
                    transforms.Resize((scale,scale)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])}
    
    return data_transforms


def feature_model_data_transforms(mode="train"):
    if mode=="train":
        data_transforms = {'train': transforms.Compose([
                    transforms.ToTensor()]),
                    'val': transforms.Compose([
                    transforms.ToTensor()
                    ])}
    if mode=="test":
        data_transforms = {'test': transforms.Compose([
                    transforms.ToTensor()])}
    if mode=="eval":
        data_transforms = {'eval': transforms.Compose([
                    transforms.ToTensor()])}
    
    return data_transforms



# Data loading

# Class MyImageFolder replaces ImageFolder for 'eval' mode!
class MyImageFolder(datasets.ImageFolder):
            def __getitem__(self, index):
                print('iterating myimagefolder')
                return super(MyImageFolder, self).__getitem__(index), self.imgs[index] # return image path
    
# Data loading function


# Class MyImageFolder replaces ImageFolder for 'eval' mode!

def load_data(data_dir, data_transforms, mode = 'eval', 
              batch_size=16, num_workers=None):
    
    if mode == 'eval':
        print('step1')
        if num_workers==None:
            num_workers=0
        else:
            pass
        # Note that in this case, data_dir does not have the train and test subfolders
           
        image_datasets = {'eval': MyImageFolder(data_dir, data_transforms['eval'])}
        #print('got image_datasets')
        #print(image_datasets)
        dataloaders = {'eval': torch.utils.data.DataLoader(image_datasets['eval'], batch_size=batch_size,shuffle=False, num_workers=num_workers) }
        #print('step 3')
        print(dataloaders)
        dataset_sizes = {'eval': len(image_datasets['eval'])}
        print(dataset_sizes)
        class_names = image_datasets['eval'].classes
        print(class_names)
        dict_c2i=image_datasets['eval'].class_to_idx
        print(dict_c2i)

    elif mode == 'test':
        if num_workers==None:
            num_workers=0
        else:
            pass

        image_datasets = {'test': datasets.ImageFolder(data_dir, data_transforms['test'])}
        dataloaders = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,shuffle=False, num_workers=num_workers) }
        dataset_sizes = {'test': len(image_datasets['test'])}
        class_names = image_datasets['test'].classes
        dict_c2i=image_datasets['test'].class_to_idx
        
    elif mode == 'train':
        if num_workers==None:
            num_workers=0
        else:
            pass
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
        dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers), 
                       'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        dict_c2i=image_datasets['train'].class_to_idx
        
    return dataloaders, dataset_sizes, class_names, dict_c2i   #dict_c2i is the dictionary of class to index

        

        


"""""""""""""""""""""""""""""""""
Feature extraction data output setup

"""""""""""""""""""""""""""""""""

def feature_eval_prep(class_to_idx, feat_path="features"):

    feat_path="features"
    # Please delete features folder if it previously exists
    if os.path.isdir(feat_path)==False:
        os.mkdir(feat_path)
    else:
        sys.exit("Please rename or delete the previous "+feat_path+" folder for new run!")
    class_names=list(class_to_idx.keys())
    for class_name in class_names:
        os.mkdir(os.path.join(feat_path, class_name))
    # Get dict_i2c to be used later in getting class names of features
    dict_c2i=class_to_idx
    class_names, idx=zip(*dict_c2i.items())
    dict_i2c=dict(zip(idx, class_names))
    
    return dict_i2c


"""""""""""""""""""""""""""""""""
Rearranging files under source folder (data_folder) with class subfolders 
into train/val/test folders under target fold (targ_folder)

"""""""""""""""""""""""""""""""""
def refolder(data_folder, targ_folder, train_fraction=0.8, val_fraction=0.2, test_fraction=0.0, 
              remove_original=False):
    r=data_folder
    classes=[f for f in os.listdir(r) if os.path.isdir(os.path.join(r,f))]
    print('1 step')
    if os.path.isdir(targ_folder):
        shutil.rmtree(targ_folder)
    os.mkdir(targ_folder)
    print('step 2')
    sub_folder=os.path.join(targ_folder, 'train')
    os.mkdir(sub_folder)
    for c in classes:
        os.mkdir(os.path.join(sub_folder,c))
    
    sub_folder=os.path.join(targ_folder, 'val')
    os.mkdir(sub_folder)
    for c in classes:
        os.mkdir(os.path.join(sub_folder,c))

    if test_fraction!=0:
        sub_folder=os.path.join(targ_folder, 'test')
        os.mkdir(sub_folder)
        for c in classes:
            os.mkdir(os.path.join(sub_folder,c))
    
    for c in classes:
        files=glob.glob(os.path.join(r,c,"*"))
        random.shuffle(files)
        train_n=int(len(files)*train_fraction)
        for f in files[:train_n]:
            filename = os.path.basename(f)
            copyfile(f, os.path.join(targ_folder,'train', c,filename))
        
        if test_fraction==0:
            for f in files[train_n:]:
                filename = os.path.basename(f)
                copyfile(f, os.path.join(targ_folder,'val', c,filename))
        
        elif test_fraction!=0:
            val_n=int(len(files)*val_fraction)
            for f in files[train_n:(train_n+val_n)]:
                filename = os.path.basename(f)
                copyfile(f, os.path.join(targ_folder,'val', c,filename))
            for f in files[(train_n+val_n):]:
                filename = os.path.basename(f)
                copyfile(f, os.path.join(targ_folder,'test', c,filename))
        
        if remove_original==True:
            shutil.rmtree(data_folder)

        

"""""""""""""""""""""""""""""""""
Feature extraction model running 

"""""""""""""""""""""""""""""""""

##### Note: each img can only have 1 label for the following code

def eval_model(model, dataloaders, dataset_sizes,  use_gpu, batch_size, dict_i2c, feat_path='features'):
    print("===>Test begains...")
    #since = time.time()
    phase='eval'
    model.eval()
    #running_corrects = 0.0
    #out_list=[]
    out_arr=[]
    # Iterate over data
    i=0 
    # Create feature path folder
    if os.path.isdir(feat_path)==False:
        print('Make directory and class subdirectories for the output data!')
        return
    for data in tqdm(dataloaders[phase]):
        # get the inputs
        (inputs, labels), (paths, _) = data
        
        
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            #labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
        print('inputs:')
        print(type(inputs))

        # forward
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        #outputs=F.softmax(outputs,dim=1) 
        outputs=np.array(outputs.data.cpu())
        #outputs=outputs.sum(axis=0)
        labels=np.array(labels)
        for j in range(len(labels)):
            path=os.path.join(feat_path, dict_i2c[labels[j]], ntpath.basename(paths[j]))
                #np.save(str(i*batch_size+j),outputs[j,:,:,:])
            np.save(path,outputs[j,:,:,:])
        i=i+1

    
    print("output shape of last batch is: {}".format(outputs.shape))
    



"""""""""""""""""""""""""""""""""
Testing model 

"""""""""""""""""""""""""""""""""
def test_model(model, dataloaders, dataset_sizes,  use_gpu):
    print("===>Test begains...")
    since = time.time()
    phase='test'

    running_corrects = 0.0
    
    # Iterate over data.
    for data in tqdm(dataloaders[phase]):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        _, preds = torch.max(outputs.data, 1)

        running_corrects += preds.eq(labels).sum().item() 

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    acc = 100.* running_corrects / dataset_sizes[phase]
    print('Test Acc: {:.4f}'.format(acc))
    


"""""""""""""""""""""""""""""""""
Training model 

"""""""""""""""""""""""""""""""""
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
                use_gpu, num_epochs, output_dir, mixup = False, alpha = 0.1):
    #print("MIXUP".format(mixup))
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                #augementation using mixup
                mixup=0
                if phase == 'train' and mixup:
                    #inputs = mixup_batch(inputs, alpha)
                    pass
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                #running_loss += loss.data[0]
                running_loss += loss.item()
                running_corrects += preds.eq(labels).sum().item() 
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model=copy.deepcopy(model)
                #best_model_wts = model.state_dict()
                
                print("Model Saving...")
                #state = {'net': model.state_dict()}
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                torch.save(model, os.path.join(output_dir,'trained_model.t7'))
                print(r"Model Saved...")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model.state_dict())
    return model







"""""""""""""""""""""""""""""""""
Training truncated model with only higher layers 

"""""""""""""""""""""""""""""""""

