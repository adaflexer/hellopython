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

import torch
import random


from torch.utils.data import DataLoader
from collections import OrderedDict

import time
from torch import nn, optim
from PIL import Image

from skimage.transform import resize
import seaborn as sns

from input_args import args_input
import predict

def main():
    
    in_arg = args_input()
    print('Just testing')
    print(in_arg)
    
     #check_command_line_arguments(in_arg)
    
    #load, build ,train, check
    dataset, dataloader= get_data_set_loader(in_arg.flower_dir)
    model, optimizer=build_model(in_arg.arc, in_arg.hidden_units)

    train_model(model, optimizer, dataloader,in_arg.epochs, in_arg.gpu, in_arg.save_dir, in_arg.learning_rate)
    save_model_to_chkpnt( dataset, model, optimizer, in_arg.epochs, in_arg.gpu, in_arg.save_dir)
    #cat_to_name= get_cat_to_name_dict(in_arg.category_names)
    #image_path=get_image_path(in_arg.image_path)

    #predict(image_path,model, in_arg.gpu, in_arg.top_k)
    
    #model_chk, optimizier_chk = load_model()
    #predict(model_chk,optimizier_chk )


   
def get_data_set_loader(data_dir) :
    
    #data_dir = in_arg['flower_dir']
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    mean=[0.485,0.456,.0406]
    sdt_dvt=[0.22,0.224,0.225]

    # TODO: Define your transforms for the training, validation, and testing sets
    # random scaling, cropping and flipping
    train_transforms = transforms.Compose([
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(50),
                                          transforms.ToTensor(),
                                        transforms.Normalize(mean,sdt_dvt)

                                         ])
    # no scaling or rotation, do resize and crop
    test_transforms= train_transforms = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,sdt_dvt),

                                        ])

    # no scaling or rotation, do resize and crop
    valid_transform=train_transforms = transforms.Compose([
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean,sdt_dvt),


                                       ])

    # TODO: Load the datasets with ImageFolder
    train_set= datasets.ImageFolder(train_dir,transform=train_transforms)

    test_set=datasets.ImageFolder(test_dir,transform=test_transforms)

    valid_set=datasets.ImageFolder(valid_dir,transform=valid_transform)

    setofdata=[train_set,test_set,valid_set]

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    testloader = DataLoader(test_set , batch_size=64, shuffle=True)
    trainloader= DataLoader(train_set, batch_size=64, shuffle=True)
    validloader= DataLoader(valid_set,batch_size=64, shuffle=True)

    dataloader=[testloader,trainloader,validloader]



    
        
    return setofdata, dataloader   





def build_model(arch, hidden_units):
    
    
  
    
    
    if arch=='vgg16':
        model_vgg=models.vgg16(pretrained=True)
        input_size=25088
    elif arch=='alexnet':
        #I had user everywhere as parameter name model_vgg. I see now i have to offer at least 2 models. Refactoring would take time. So because of this im not changing parameter name
        model_vgg=models.alexnet(pretrained=True)
        input_size=9216

        
    for param in model_vgg.parameters():
         param.requires_grad=False 
    
    new_classifier = nn.Sequential (OrderedDict([
                                 ('fc1', nn.Linear(input_size ,hidden_units)) , 
                                 ('relu',nn.ReLU()), 
                                 ('dropout',nn.Dropout(0.2)), 
                                 ('fc2', nn.Linear(hidden_units,102)) ,                      
                                 ('output', nn.LogSoftmax (dim = 1))
                          ]))

    # replace the existing classifier with new one
    model_vgg.classifier=new_classifier
    
    optimizer = optim.Adam(model_vgg.classifier.parameters(), lr=0.001)

    return model_vgg, optimizer
    
   

def train_model(model_vgg,optimizer,dataloader, arg_epochs, gpu, save_dir, arg_learning_rate):
        
    
    trainloader=dataloader[1]
    validloader=dataloader[2]
    #epochs = 1
    criterion = nn.NLLLoss()
    if (gpu):
       if torch.cuda.is_available():
           device= torch.device("cuda")
           print('CUDA is avlb')
       else:
            print("GPU requested for Train but is not avlb!")
            exit()      
    else:
       device= torch.device("cpu") 

    print(device)          
# Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model_vgg.classifier.parameters(), lr=arg_learning_rate)
    #device = torch.cuda.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_vgg.to(device)
    
    print_every = 80
    steps = 0
    running_loss=0
    start_time= time.time()
    epochs=arg_epochs
    
    



    for e in range(epochs):
   
        for images, labels in iter(trainloader):
            images, labels = images.to(device), labels.to(device)
            steps += 1
        
            print(steps)
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            output = model_vgg.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item()
        
            if steps % print_every==0:
                valid_loss=0
                accuracy=0
                # Make sure network is in eval mode for inference
                model_vgg.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images,labels=images.to(device),labels.to(device)
                        output=model_vgg.forward(images)
                        batch_loss=criterion(output,labels)
                        valid_loss+=batch_loss.item()
                        #probabilities
                        ps=torch.exp(output)
                    
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                    
                    
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))        
                    running_loss = 0    
                    model_vgg.train()
                    
        
def save_model_to_chkpnt(data_sets,model_vgg, optimizer,device, epochs, save_dir):
    
    train_set=data_sets[0]
    model_vgg.class_to_idx = train_set.class_to_idx
    print(model_vgg.class_to_idx)
    checkpoint = {'epochs': epochs,
              'arch': 'vgg16',
              'state_dict': model_vgg.state_dict(),
              'classifier': model_vgg.classifier,
              'model_state': model_vgg.state_dict(),
              'optimizer': optimizer.state_dict(),              
              'class_to_idx': model_vgg.class_to_idx}
    torch.save(checkpoint, save_dir +'/'+ 'checkpoint.pth')
    print('model saved to '+' ' +save_dir +'/')
    
    



if __name__=="__main__":
    main()    