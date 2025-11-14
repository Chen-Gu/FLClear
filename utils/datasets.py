import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from utils.utils import *
from torchvision.datasets import ImageFolder

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class VectorAugment(Dataset):
    def __init__(self, wm_dataset, wm_idx, dim, num=10, key=None):
        if len(wm_idx) != 1:
            raise ValueError("wm_idx must contain exactly one index")
        self.wm_dataset = wm_dataset
        self.dim = dim
        self.w_idx = list(wm_idx)[0]  
        self.wm,  self.key_vec, self.seed = self.wm_dataset[self.w_idx]
 
        self.num = num
        if self.wm is None or self.key_vec is None:
            raise ValueError("Invalid wm or key from wm_dataset")
        self.sigma_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.5,0.7,0.9,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
        self.vectors = self._generate_samples()
       
    def _generate_samples(self):

        noise = []
        for scale in self.sigma_list:        
            temp = torch.normal(mean=0, std=scale, size=(self.num, self.dim))
            noise.append(temp)
        noise = torch.cat(noise, dim=0) 
        vector_sets = torch.cat((self.key_vec + noise, self.key_vec.unsqueeze(0)), dim=0).to(dtype=torch.float32)  

        return vector_sets

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, item):
        return self.vectors[item]



class WMDataset(Dataset):
    def __init__(self, image_dir, dim, transform=None):
        
        self.root_dir = image_dir
        self.transform = transform
        self.dim = dim
       
        self.image_folder = ImageFolder(image_dir, transform=None)  
        self.seeds = self.image_folder.classes 

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        
        image, _ = self.image_folder[idx]  
        seed = self.seeds[self.image_folder.targets[idx]]  

       
        key = generate_hash_code(seed=seed, dim=self.dim)  
        key = torch.tensor(key, dtype=torch.float32)  
       
        if self.transform:
            image = self.transform(image)

        return image, key, seed

