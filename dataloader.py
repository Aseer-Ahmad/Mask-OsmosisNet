#dataloader.py
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

class BSDS300Dataset(Dataset):

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

        img_pth = os.path.join(self.root_dir, self.split_type,
                                str(self.data_list[idx])+'.jpg')

        image = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE) 
        tensor = torch.tensor(image, dtype = torch.float64).unsqueeze(0)

        # transforms
        tensor = transforms.Resize((self.img_size, self.img_size))(tensor)     

        return tensor
    