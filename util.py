import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import json

def get_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(225), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 102, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 102, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 102, shuffle = True)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return trainloader, validloader, testloader, train_datasets        
        
        
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    if img.size[0]>img.size[1]:
        img.resize((10000, 256))
    else:
        img.resize((256, 10000))
    
    width, height = img.size
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    
    np_img = np.array(img)/255
    
    np_img = (np_img-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
    
    np_img = np_img.transpose((2, 0, 1))
    
    return np_img

