#!/usr/bin/env python3
import yaml
from train import ModelTrainer
from CustomDataset import MaskDataset
import torch
from MaskModel.unet_model import UNet
from torchsummary import summary
import os
import shutil
import sys
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import IterableDataset
from datasets import Dataset

CONFIG_YAML = 'config.yaml'

def read_config(file_path):
    """
    Reads a YAML configuration file and returns the content as a Python dictionary.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The configuration as a dictionary.
    """
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

class HFDatasetWrapper(IterableDataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __iter__(self):
        for item in self.hf_dataset:
            yield item

def getMaskDataset(config):
    """
    Create and return Mask Dataset objects for training and testing .

    Parameters:
    config (dict): A dictionary containing configuration parameters for dataset creation.

    Returns:
    tuple: A tuple containing two Datasets objects:
        - train_dataset: Datasets for the training dataset.
        - test_dataset: Datasets for the testing dataset.
    """
    # train_dataset = MaskDataset(config['TRAIN_FILENAME'], 
    #                          config['ROOT_DIR'], 
    #                          "train", 
    #                          config['IMG_SIZE'])
    
    # test_dataset = MaskDataset(config['TEST_FILENAME'], 
    #                          config['ROOT_DIR'], 
    #                          "test", 
    #                          config['IMG_SIZE'])
    
    transform = transforms.Compose([
                    transforms.Resize((config['IMG_SIZE'], config['IMG_SIZE']), antialias = True),
                    transforms.Normalize(mean = [0.44531356896770125], std = [0.2692461874154524]),
                    transforms.ToTensor(),
                ])
    
    def apply_transforms(example):
        example['image'] = transform(example['image'])  # Apply transform to the image field
        return example

    # train_dataset = load_dataset("aseeransari/ImageNet-Sampled", split="temp")
    # test_dataset  = load_dataset("aseeransari/ImageNet-Sampled", split="temp")    
    
    test_dataset  = load_dataset("aseeransari/ImageNet-Sampled", split="test")
    train_dataset = load_dataset("aseeransari/ImageNet-Sampled", split="train")

    test_dataset  = test_dataset.map(apply_transforms, batched=True)
    train_dataset = train_dataset.map(apply_transforms, batched=True)

    # train_dataset = Dataset.from_list(list(train_dataset))
    # test_dataset  = Dataset.from_list(list(test_dataset))
    # test_dataset = test_dataset.with_format("torch")
    # train_dataset = train_dataset.with_format("torch")

    return (train_dataset, test_dataset)

def main(config):

    # make experiment directory
    exp_path = os.path.join(config["OUTPUT_DIR"], config["EXP_NAME"])
    img_pth  = os.path.join(config["OUTPUT_DIR"], config["EXP_NAME"], "imgs")
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
        os.makedirs(img_pth)
        
    # to print to a file
    # sys.stdout = open(os.path.join(exp_path, 'output.txt'),'wt')

    print(f"CONFIG : \n{config}\n")

    shutil.copyfile(CONFIG_YAML, os.path.join(exp_path, CONFIG_YAML))

    # get train , test Dataset classes
    train_dataset, test_dataset = getMaskDataset(config)
    print(f"train test dataset loaded")
    print(f"train size : {train_dataset.__len__()}")
    print(f"test  size  : {test_dataset.__len__()}")
    
    # get model based on inp and out channels
    model = UNet(config['INP_CHANNELS'], config['OUT_CHANNELS'])
    print(f"model loaded")
    
    #print model summary
    print(f"model summary")
    model_sum = summary(model, input_size=(config['INP_CHANNELS'], config['IMG_SIZE'], config['IMG_SIZE']), verbose =0)
    model_sum = str(model_sum).encode('ascii', errors='replace')
    print(model_sum.decode())

    # configure model trainer 
    trainer = ModelTrainer(
        output_dir = exp_path,
        optimizer= config['OPT'],
        scheduler= config['SCHEDL'],
        lr = config['LR'],
        weight_decay= config['WEIGHT_DECAY'], 
        train_batch_size = config['TRAIN_BATCH'],
        test_batch_size = config['TEST_BATCH'],
    )
    print(f"trainer configurations set")

    trainer.train(
        model = model,
        epochs = config['EPOCHS'],
        alpha1 = config['ALPHA1'],
        alpha2 = config['ALPHA2'],
        mask_density = config['MASK_DEN'],
        resume_checkpoint_file = config['RESUME_CHECKPOINT'],
        save_every = config['SAVE_EVERY_ITER'], 
        batch_plot_every = config['BATCH_PLOT_EVERY_ITER'],
        val_every = config['VAL_EVERY_EPOCH'],
        train_dataset = train_dataset , 
        test_dataset = test_dataset
    )
    

if __name__ == '__main__':
    config = read_config(CONFIG_YAML)
    main(config)