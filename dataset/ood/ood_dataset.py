import glob
import os
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image
from tools.function import get_pkl_rootpath
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torch

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ood_dataloader(data.Dataset):

    def __init__(self, dataset_name, datapath=None, transform=None):
        self.root_path = datapath
        self.transform = transform
        # Images must have the format [idx].png, their labels have the format [idx].txt
        self.img_id = [f for f os.listdir(self.root_path) if f.endswith('.png')]
        self.img_idx = [int(imgname.split('.')[0]) for imgname in self.img_id]
        self.label = []
        for idx in self.img_idx:
            label_file = os.path.join(self.root_path, str(idx) + '.txt')
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    self.label.append(int(f.read().strip()))
        # Either there should be one label for each image, or no labels (for the test set)
        if len(self.label) == 0:
            self.label = np.zeros(len(self.img_id))
        assert(len(self.img_id) == len(self.label) == len(self.img_idx))
        self.attr_num = len(np.unique(self.label))
                
    def __getitem__(self, index):
        
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        
        img = Image.open(imgpath)
        
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            img = PIL.Image.merge("RGB", (r,g,b))
        elif img.mode != 'RGB':
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
            
        return img, gt_label

    def __len__(self):
        return self.img_num
