# main.py
import yaml
from train import ModelTrainer
from dataloader import BSDS300Dataset

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

def getDataSets(config):
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
    
    return (train_dataset, test_dataset)

def main():
    config = read_config(CONFIG_YAML)
    print(config)

    train_dataset, test_dataset = getDataSets(config)

	trainer = ModelTrainer(
		output_dir= ,
        optimizer= ,
        scheduler= ,
		learning_rate= ,
		weight_decay= , 
		train_batch_size  = ,
		test_batch_size  = 
	)

    trainer.train(
		epochs = ,
        resume_checkpoint_file = ,
        save_every =, 
        train_dataset = train_dataset, 
        test_dataseet = test_dataset
    )
    




if __name__ == '__main__':
    main()