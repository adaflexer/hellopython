import numpy as np
import os
from os.path import isfile
from os.path import join

import json
import helper

import torch
import torchvision
import torch.nn.functional as F

from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

import random


from torch.utils.data import DataLoader
from collections import OrderedDict

import time
from torch import nn, optim
from PIL import Image

from skimage.transform import resize
import seaborn as sns
from input_args import args_input



def main():
    
    in_arg = args_input()
    
    print(in_arg)

    predict(in_arg.image_path, in_arg.gpu, in_arg.topk, in_arg.checkpoint)

    cat_to_name= get_cat_to_name_dict(in_arg.category_names)



def load_model(filepath):
        
    checkpoint = torch.load(filepath+ '/'+ 'checkpoint.pth')
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    optimizer=optim.Adam(model.classifier.parameters(), lr=0.001)
    model.arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    # classifier has the correct hyperparameter  
    
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs=checkpoint['epochs']
    optimizer.load_state_dict(checkpoint['optimizer'])
    print(model)
    
    return model, optimizer

def get_cat_to_name_dict(category_names):
   
    
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
        return cat_to_name


def predict(image_path, gpu_request, topk, checkpoint_path):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    checkpoint_path = checkpoint_path + '/'+ 'checkpoint.pth'
    model_vgg, optimizier = load_model(checkpoint_path)

    if gpu_request:
        if torch.cuda.is_available(): 

            device= torch.device("cuda:0")
        else:

            device=torch.device("cpu")   

            print('GPU requested but not available')
            exit()
    else :   
        if torch.cuda.is_available(): 

            device= torch.device("cuda:0")
        else :

            device=torch.device("cpu")    


    model_vgg.to(device)
    print(device)
    model_vgg.eval()
    
    with torch.no_grad():
        im=process_image(image_path)
        
        #create tensor
        
        im=torch.from_numpy(np.array([im])).float()
        im,model=im.to(device), model_vgg.to(device)
        output=model_vgg.forward(im)
        probalities=torch.exp(output)
        print(probalities)
        # this method returns both the highest k probabilities and the indices of those probabilities corresponding to the classes.
        probs, indices= probalities.topk(topk)
        #dict mapping (name of class to label)
        class_to_idx = model.class_to_idx
        # Make sure to invert the dictionary so you get a mapping from index to class as well.
        class_to_idx_inverted={v:k for k,v in class_to_idx.items()}
        print(class_to_idx_inverted)
        print(indices)
        print(probs)
        
        probs=probs.reshape(-1)
        print(probs)
        
        #flatten tensor
        indices =indices.reshape(-1)
        print(indices)
        indices=indices.tolist()
        print(indices)
        
        classes=[]
       
        
        
        #common practice is to predict the top 5 or so (usually called top- 𝐾 ) most probable classes
        for idx in indices:
            print(idx)
            classes.append(class_to_idx_inverted[idx])
            print(class_to_idx_inverted[idx])
            
        
   

    
    print(probs)
    print(classes)
    return probs, classes


def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        im= Image.open(image)
        im= im.resize((256,256))
        mean= np.array ([0.485, 0.456, 0.406])
        std_dvt= np.array([0.229,0.224,0.225])
        
    #https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=crop#PIL.Image.Image.crop   
    #This crops the input image with the provided coordinates:  left(256-224)/2 , up =(256-224)/2,right, lower
        cropped_widght= 256-224
        cropped_height=226-224
        im=im.crop((cropped_widght/2, cropped_height/2, ((cropped_widght/2)+224), ((cropped_height/2)+224)))
        im=np.array(im)/255
        im=(im-mean)/std_dvt
        im = im.transpose((2,0,1))
        return im

def imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        return ax


if __name__=="__main__":
    main()    