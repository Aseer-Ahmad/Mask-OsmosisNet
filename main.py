# main.py
import yaml
from train import ModelTrainer
from CustomDataset import BSDS300Dataset
from MaskModel.unet import UNet

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
    Create and return Datasets objects for training and testing .

    Parameters:
    config (dict): A dictionary containing configuration parameters for dataset creation.

    Returns:
    tuple: A tuple containing two Datasets objects:
        - train_dataset: Datasets for the training dataset.
        - test_dataset: Datasets for the testing dataset.
    """
    train_dataset = BSDS300Dataset(config['TRAIN_FILENAME'], 
                             config['ROOT_DIR'], 
                             "train", 
                             config['IMG_SIZE'])
    
    test_dataset = BSDS300Dataset(config['TEST_FILENAME'], 
                             config['ROOT_DIR'], 
                             "test", 
                             config['IMG_SIZE'])
    
    return (train_dataset, test_dataset)

def main(config):

    # get train , test Dataset classes
    train_dataset, test_dataset = getDataSets(config)
    print(f"train test dataset loaded")
    print(f"train size : {len(train_dataset)}")
    print(f"test  size : {len(test_dataset)}")
    
    # get model based on inp and out channels
    model = UNet(config['INP_CHANNELS'], config['OUT_CHANNELS'])
    print(f"model loaded")
    print(model)
    # configure model trainer 
    trainer = ModelTrainer(
        output_dir= config['OUTPUT_DIR'],
        optimizer= config['OPT'],
        scheduler= config['SCHEDL'],
        lr= config['LR'],
        weight_decay= config['WEIGHT_DECAY'], 
        train_batch_size = config['TRAIN_BATCH'],
        test_batch_size = config['TEST_BATCH']
    )
    print(f"trainer configurations set")
    print(f"CONFIG : \n{config}\n")

    trainer.train(
        model = model,
        epochs = config['EPOCHS'],
        resume_checkpoint_file = config['RESUME_CHECKPOINT'],
        save_every = config['SAVE_EVERY'], 
        val_every = config['VAL_EVERY'],
        train_dataset = train_dataset , 
        test_dataset = test_dataset
    )
    

if __name__ == '__main__':
    config = read_config(CONFIG_YAML)
    main(config)