# main.py
from dataloader import BSDS300Dataset
from torch.utils.data import DataLoader
import yaml

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

def getDataLoaders(config):
    """
    Create and return DataLoader objects for training and testing datasets.

    Parameters:
    config (dict): A dictionary containing configuration parameters for dataset and dataloader creation.

    Returns:
    tuple: A tuple containing two DataLoader objects:
        - train_dataloader: DataLoader for the training dataset.
        - test_dataloader: DataLoader for the testing dataset.
    """
    train_dataset = BSDS300Dataset(config["TRAIN_FILENAME"], 
                             config["ROOT_DIR"], 
                             "train", 
                             config["IMG_SIZE"])
    test_dataset = BSDS300Dataset(config["TEST_FILENAME"], 
                             config["ROOT_DIR"], 
                             "test", 
                             config["IMG_SIZE"])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)

    return (train_dataloader, test_dataloader)

def main():
    config = read_config(CONFIG_YAML)
    print(config)


    train_dataloader, test_dataloader = getDataLoaders(config)






if __name__ == '__main__':
    main()