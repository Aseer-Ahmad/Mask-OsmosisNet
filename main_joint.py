#!/usr/bin/env python3
import yaml
from train import JointModelTrainer
from CustomDataset import MaskDataset
import torch
# from MaskModel.GCtx_UNet import GCViT_Unet as ViT_seg
# from MaskModel.config_GCtx import get_config
from MaskModel.MaskNet import MaskNet
from MaskModel.InpaintingNet import InpaintingNet
from MaskModel.unet import UNet
from torchsummary import summary
import os
import shutil
import sys
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import IterableDataset
from datasets import Dataset
from datasets import disable_caching
disable_caching()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_LOGS"]="+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"]="1"

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

class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]['image']

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
    
    # test_dataset  = load_dataset("aseeransari/ImageNet-Sampled", split="test")
    # train_dataset = load_dataset("aseeransari/ImageNet-Sampled", split="train")
    # test_dataset = test_dataset.remove_columns(['file_name'])
    # train_dataset = train_dataset.remove_columns(['file_name'])

    test_dataset  = load_dataset("aseeransari/BSDS", split="test")
    train_dataset = load_dataset("aseeransari/BSDS", split="train")
    test_dataset = test_dataset.remove_columns(['file_name'])
    train_dataset = train_dataset.remove_columns(['file_name'])

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
    


    # get Mask model

    # config_ViT = get_config()
    # maskNet = ViT_seg(config_ViT, img_size=128, num_classes=1)
    # print(f"Mask model loaded")
    # print(f"Mask model summary")
    # model_sum = summary(maskNet, 
    #                     input_data =(1, 128, 128), 
    #                     col_names=["kernel_size", "output_size", "num_params", "mult_adds"])

    maskNet = MaskNet(config['MN_INP_CHANNELS'], config['MN_OUT_CHANNELS'], tar_den = config['MASK_DEN'])
    model_sum = summary(maskNet, 
                        input_data =(config['MN_INP_CHANNELS'], config['IMG_SIZE'], config['IMG_SIZE']), 
                        col_names=["kernel_size", "output_size", "num_params", "mult_adds"])
    
    model_sum = str(model_sum).encode('ascii', errors='replace')
    print(model_sum.decode())

    # get Inpainting model
    inpNet = InpaintingNet(config['IN_INP_CHANNELS'], config['IN_OUT_CHANNELS'])
    print(f"Inpainting model loaded")
    print(f"Inpainting model summary")
    model_sum = summary(inpNet, 
                        input_data=(config['IN_INP_CHANNELS'], config['IMG_SIZE'], config['IMG_SIZE']),
                        col_names=["kernel_size", "output_size", "num_params", "mult_adds"])
    model_sum = str(model_sum).encode('ascii', errors='replace')
    print(model_sum.decode())



    # configure model trainer 
    trainer = JointModelTrainer(
        output_dir = exp_path,
        opt1= config['OPT1'],
        opt2= config['OPT2'],
        scheduler1= config['SCHEDL1'],
        scheduler2= config['SCHEDL2'],
        train_batch_size = config['TRAIN_BATCH'],
        test_batch_size = config['TEST_BATCH'],
    )
    
    print(f"trainer configurations set")

    trainer.train(
        maskModel = maskNet,
        inpModel  = inpNet,
        epochs = config['EPOCHS'],
        alpha1 = config['ALPHA1'],
        alpha2 = config['ALPHA2'],
        offset = config['RES_LOSS_OFFSET'],
        tau = config['TAU'], 
        mask_density = config['MASK_DEN'],
        img_size = config['IMG_SIZE'],
        model_1_ckp_file = config['RESUME_CHECKPOINT_MN'],
        model_2_ckp_file = config['RESUME_CHECKPOINT_IN'],
        save_every = config['SAVE_EVERY_ITER'], 
        batch_plot_every = config['BATCH_PLOT_EVERY_ITER'],
        val_every = config['VAL_EVERY_ITER'],
        skip_norm = config['SKIP_NORM'],
        max_norm  = config['MAX_NORM'],
        train_dataset = train_dataset , 
        test_dataset = test_dataset,
        solver = config['SOLVER_TYPE'],

    )
    
if __name__ == '__main__':
    config = read_config(CONFIG_YAML)
    main(config)
