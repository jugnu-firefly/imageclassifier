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

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", action = "store", default = "flowers")
parser.add_argument("--save_dir", action = "store", default = 'checkpoint.pth')
parser.add_argument("--arch", action = "store", default = "vgg16")
parser.add_argument('--learning_rate', action="store", type=float,default=0.01)
parser.add_argument('--hidden_units', action="store", type=int, default=512)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--gpu', action="store", default="gpu")
args = parser.parse_args()

def train():
    trainloader, validloader, testloader, train_datasets = util.get_data(args.data_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == "gpu" else "cpu")
    
    model, criterion, optimizer = model.load_model(args.arch, args.learning_rate, args.hidden_units, args.gpu)
    model.to(device);
    epochs = args.epochs
    print_every = 5
    steps = 0
    train_loss=0
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels= inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            train_loss+=loss.item()
        
            accuracy = 0
            valid_loss = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                
                    valid_loss += batch_loss.item()
                
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss:{train_loss:.3f}.."
                  f"Validation loss:{valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            train_loss = 0
            model.train()
            
            
            
    # TODO: Save the checkpoint 
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'input_size': 25088, 'output_size': 102, 'classifier': model.classifier, 'epochs' : epochs, 'state_dict': model.state_dict(), 'optimizer_dict': optimizer.state_dict(), 'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, args.save_dir)