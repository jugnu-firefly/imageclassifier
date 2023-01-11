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
import model
import train

parser = argparse.ArgumentParser()
parser.add_argument("image_path", action = "store", default = "flowers")
parser.add_argument("checkpoint", action = "store", default = 'checkpoint.pth')
parser.add_argument("--top_k", action = "store", default = 5, type = int)
parser.add_argument("--category_names", action = "store", default = 'cat_to_name.json')
parser.add_argument("--gpu", action = "store", default = 'gpu')
args = parser.parse_args()

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == "gpu" else "cpu")
    
    with open(args.category_names, 'r') as json_file:
        cat_name = json.load(json_file)
        
    model = model.load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    img = process_image(ar.image_path)
    img = torch.from_numpy(np.array([img])).float()  
    input = img.to(device)
    
    with torch.no_grad():
        logps = model.forward(input)
        ps = torch.exp(logps)
                
    model.train()
    probs, classes = ps.topk(args.top_k)
    probability = np.array(classes[0])
    
    y = []
    for i in probability:
        j = str(i+1)
        if j in cat_name:
            y.append(cat_name[j])
            
    print(y)
    print(probs[0])