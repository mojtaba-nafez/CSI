import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from utils_.utils import set_random_seed
from datasets.cutpast_transformation import *
from PIL import Image
from glob import glob
import pickle
import random
import rasterio
import re
from torchvision.datasets.folder import default_loader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torchvision
import subprocess
from tqdm import tqdm
import requests
import shutil 
from PIL import Image
import shutil
import random
import zipfile
import time

CLASS_NAMES = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut', 'hazelnut', 'screw', 'carpet', 'leather', 'cable']
DATA_PATH = './data/'
class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)

class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        print("category MVTecDataset:", category)
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1
        return image, target

    def __len__(self):
        return len(self.image_files)

class MVTecDataset_Cutpasted(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        print("category MVTecDataset_Cutpasted:", category)
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count!=-1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, -1

    def __len__(self):
        return len(self.image_files)
    

class HEAD_CT_DATASET(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]
    
    def __len__(self):
        return len(self.image_files)

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

def sparse2coarse(targets):
    coarse_labels = np.array(
        [4,1,14, 8, 0, 6, 7, 7, 18, 3, 3,
         14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5,
         10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10,
         12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15,
         13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0,
         17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12,
         1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13,
         15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8,
         19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,])
    return coarse_labels[targets]

class CIFAR_CORRUCPION(Dataset):
    def __init__(self, transform=None, normal_idx = [0], cifar_corruption_label = 'CIFAR-10-C/labels.npy', cifar_corruption_data = './CIFAR-10-C/defocus_blur.npy'):
        self.labels_10 = np.load(cifar_corruption_label)
        self.labels_10 = self.labels_10[:10000]
        if cifar_corruption_label == 'CIFAR-100-C/labels.npy':
            self.labels_10 = sparse2coarse(self.labels_10)
        self.data = np.load(cifar_corruption_data)
        self.data = self.data[:10000]
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels_10[index]
        if self.transform:
            x = Image.fromarray((x * 255).astype(np.uint8))
            x = self.transform(x)    
        return x, y
    
    def __len__(self):
        return len(self.data)

class MNIST_CORRUPTION(Dataset):
    def __init__(self, root_dir, corruption_type, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.corruption_type = corruption_type
        self.train = train

        indicator = 'train' if train else 'test'
        folder = os.path.join(self.root_dir, self.corruption_type, f'saved_{indicator}_images')
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        
        if train:
            data = np.load(os.path.join(root_dir, corruption_type, 'train_images.npy'))
            labels = np.load(os.path.join(root_dir, corruption_type, 'train_labels.npy'))
        else:
            data = np.load(os.path.join(root_dir, corruption_type, 'test_images.npy'))
            labels = np.load(os.path.join(root_dir, corruption_type, 'test_labels.npy'))
            
        self.labels = labels
        self.image_paths = []

        for idx, img in enumerate(data):
            path = os.path.join(folder, f"{idx}.png")
            self.image_paths.append(path)
            
            if not os.path.exists(path):
                img_pil = torchvision.transforms.ToPILImage()(img)
                img_pil.save(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB") 

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
   

class ISIC2018(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


class ImageNet30_Dataset(Dataset):
    def __init__(self, image_path, labels, transform=None):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


class Custom_Dataset(Dataset):
    def __init__(self, image_path, targets, transform=None):
        self.transform = transform
        self.image_files = image_path
        self.targets = targets
        
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

    def __len__(self):
        return len(self.image_files)
        

