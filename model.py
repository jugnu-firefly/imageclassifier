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
import util

def load_model(mod = "vgg16", lr, hu, on):
    device = torch.device("cuda" if torch.cuda.is_available() and on == "gpu" else "cpu")
    
    if mod == "vgg16":
        model = models.vgg16(pretrained = True)
        
    elif mod == "densenet121":
        model = models.densenet121(pretrained = True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(25088, hu), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hu, 102), nn.LogSoftmax(dim=1))
    
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    return model, criterion, optimizer

    
def load_checkpoint(filepath):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(filepath, map_location=map_location)
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
