import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

import torch.nn as nn
from typing import Tuple, Iterable

import torch.nn.utils as utils

from functorch import vmap
import torch.nn.functional as F

import matplotlib.pyplot as plt



# Siamese Model
class SiameseNetwork(nn.Module):
    def __init__(self, cnn_layers: Iterable, linear_layers):
        super().__init__()
        
        # pass in a NN
        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.linear_layers = nn.Sequential(*linear_layers)
        
    
    def forward(self, x_query: torch.Tensor, x_support: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward Pass:
            arg: Query Image (any image)
                 Support Image
        
        """
                
        x_query = self.cnn_layers(x_query)        
        x_support = self.cnn_layers(x_support)
        
        # similarity between query and support images. will be used for triplet loss
        similarity = vmap(F.pairwise_distance)(x_query, x_support)[:, None]        

        # logits for query image. will be used for cross entropy loss
        logits = self.linear_layers(x_query.view(x_query.size()[0], -1))
            
        return similarity, logits

    

def get_image(path):
    
    """ Takes in image path and returns transformed tensor image"""
    
    image = Image.open(path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image = transform(image)
        
    # adding another dimension to indicate a batch of size 1
    image = torch.unsqueeze(image, 0)
    
    return image



def get_prediction(img_path):
    
    """ Predicts if the image is of accident or not, and the severity of the accident """

    accident_model_path = 'saved_models/accident_best_model.pth'
    severity_model_path = 'saved_models/severity_best_model.pth'

    accident_model = torch.load(accident_model_path)
    severity_model = torch.load(severity_model_path)

    accident_model.eval()
    severity_model.eval()

    image = get_image(img_path)

    _, acc_logits = accident_model(image, image)
    acc_logits = torch.softmax(acc_logits, dim=-1)

    accident_prediction = torch.argmax(acc_logits, dim=-1).item()

    _, sev_logits = severity_model(image, image)
    sev_logits = torch.softmax(sev_logits, dim=-1)

    severity_prediction = torch.argmax(sev_logits, dim=-1).item()

    
    return accident_prediction, severity_prediction
