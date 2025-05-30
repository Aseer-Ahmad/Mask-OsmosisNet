#dataloader.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import cv2


class MaskDataset(Dataset):

    def __init__(self, csv_file, root_dir, split_type, img_size):
        """
        Arguments:
            csv_file (string): Path to the csv file
            root_dir (string): Directory with all the images.
            split_type (string): train / test
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data_list  = pd.read_csv(os.path.join(root_dir, csv_file), header=None)[0].values.tolist()
        self.root_dir   = root_dir
        self.split_type = split_type
        self.img_size   = img_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        img_name = str(self.data_list[idx])
        img_pth = os.path.join(self.root_dir, self.split_type,img_name)

        image = cv2.imread(img_pth) 

        # resizing ; Tobias paper cropped images
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Grayscale(),
                    transforms.Resize((self.img_size, self.img_size), antialias = True),
                    # transforms.RandomCrop((self.img_size, self.img_size))
                    # transforms.Normalize(mean = [0.], std = [1.])
                    ])
        
        tensor_transformed = transform(image)

        # https://stackoverflow.com/questions/65699020/calculate-standard-deviation-for-grayscale-imagenet-pixel-values-with-rotation-m
        # tensor_transformed = transforms.Normalize(mean = [0.44531356896770125], std = [0.2692461874154524])(tensor)
        # tensor_transformed = transforms.Normalize(mean = [0.], std = [1.])(tensor)     
        
        return (tensor_transformed, img_name)
    