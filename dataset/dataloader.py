#Import libraries
import cv2
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

def get_relevant_indicies(dataset):
    """Returns the indicies of the classes in the dataset"""
    indicies = []
    for i in range(len(dataset)):
        idx = dataset[i][1]
        indicies.append(idx)
    return indicies

def get_data(batch_size, folder):
    """Takes a batch_size and the name of the folder (name of folder most likely called dataset)
    Example:
    get_data(1, "~/aps360-proj/dataset")
    """
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    #Load images
    trainset = torchvision.datasets.ImageFolder(folder, transform=transform)    

    train_size = int(0.9*len(trainset)) #90% split for training
    val_size = len(trainset) - train_size #10% for validation
    train_split, val_split = torch.utils.data.random_split(trainset, [train_size, val_size])
    #SAMPLER DOES NOT WORK FOR SOME REASON LOL so I split and then do this
    #https://stackoverflow.com/questions/74291946/wrong-labels-when-using-dataloader-pytorch
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, num_workers=1)
    #visualize(train_loader)
    #visualize(val_loader)
    return train_loader, val_loader
def visualize(loader):
    """Visualize the data we parse, mostly for a sanity check"""
    classes = ("Normal", "Talking", "Yawning")
    k = 0
    for images, labels in loader:
        image = images[0]
        img = np.transpose(image, [1,2,0])
        # normalize pixel intensity values to [0, 1]
        img = img / 2 + 0.5
        plt.subplot(3, 5, k+1, title=classes[labels])
        plt.axis('off')
        plt.imshow(img) #Creates the image
        k += 1
        if k > 14:
            break

    plt.show() #Actually shows the image

def main():
    #To test it's working: get_data(1, "~/aps360-proj/testing")
    # f, g = get_data(1000, "~/aps360-proj/dataset"), crashes on my local test on PC
    # for i in f:
    #     print(i)
    print("Why are you not importing this file idiot")
if __name__ == '__main__':
    main()