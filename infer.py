# infer.py
import os
from MaskModel.MaskNet import MaskNet
from MaskModel.unet_model import UNet
from main import read_config
from utils import loadCheckpoint
from CustomDataset import MaskDataset
from torch.utils.data import DataLoader
from datetime import datetime
import torch
from InpaintingSolver.bi_cg import OsmosisInpainting

from utils import normalize
import cv2

class Binarize(object):

    @staticmethod
    def hard_rounding(x):
        return torch.floor(x + 0.6)


def get_model_by_task(task):
    if task == 'unet_single' :
        return UNet(1, 1, tar_den = 0.1)
    elif task == 'unet_double' :
        return UNet(1, 2, tar_den = 0.1)
    elif task == 'masknet_single' :
        return MaskNet(1, 2, tar_den = 0.1)

def infer(infer_path):
    '''
    - list of bin_method ; determines no. of output folders
    - init model for eval
    - for each test folder
        - clear the test folders output dir
        - prepare test data loaders
        - iterate over data
            - canny masks
            - BiCG solve
            - get non-binary masks
            - BiCG solve
            - for each binarization method
                - binarize masks
                - BiCG solve
                - save img sep & concat (original, canny mask -> sol, non-bin mask -> sol, bin mask -> sol)
                - save data csv
    '''

    infer_config = read_config(os.path.join(infer_path, "infer.yaml"))
    print(infer_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}\n")

    for key, task  in infer_config["TASK"].items():
        print(f"inferring for task : {key}")

        # load models
        if task['MODEL_CKP_PTH'] :
            model = get_model_by_task(task['TYPE'])
            model, _ = loadCheckpoint(model, None, task['MODEL_CKP_PTH'])
            model = model.double()
            model.to(device)
            model.eval()
            print("model loaded")
        else : 
            print(f"skipping task : {key} . No checkpoint path found.")
            continue

        # create dataloaders
        test_dataset     = MaskDataset(task['DATA_LIST'], task['DATA_PTH'], "test", 128)
        test_dataloader  = DataLoader(test_dataset, batch_size=8)
        print(f"data loaded : {len(test_dataset)}")
        
        # create directories
        f_name = task['MODEL_CKP_PTH'].split(".")[0].split("/")[-1] \
               + "_" + task["TYPE"] + "_" + task["BIN_METHOD"] 
        
        imgs_pth = os.path.join(infer_path, f_name, "images")
        
        img_size = task["IMG_SIZE"]
        offset   = 12

        # if path exist then skip
        if os.path.exists(imgs_pth):
            print(f"skipping task : {key} , PATH ALREADY EXIST")
            continue
        else:
            os.makedirs(imgs_pth)
            
        # get mask and solve
        with torch.no_grad():
            for i, (X, X_names) in enumerate(test_dataloader):

                X = X.to(device, dtype=torch.float64)
                mask  = model(X)
                
                if mask.shape[1] == 2:
                    mask1 = mask[:, 0, :, :].unsqueeze(1)
                    mask2 = mask[:, 1, :, :].unsqueeze(1)
                else:
                    mask1 = mask
                    mask2 = mask

                # binarize (0,1)
                mask1_bin = Binarize.hard_rounding(mask1)
                mask2_bin = Binarize.hard_rounding(mask2)

                # solve for canny

                # solve for non-binary masks

                # solve for binary masks
                osmosis = OsmosisInpainting(None, X, mask1_bin, mask2_bin, offset=offset, tau=30000, eps = 1e-11, device = device, apply_canny=False)
                osmosis.calculateWeights(d_verbose=False, m_verbose=False, s_verbose=False)
                save_batch = [False]
                loss3, tts, max_k, df_stencils, X_rec = osmosis.solveBatchParallel(None, None, 1, save_batch = save_batch, verbose = False)

                fname = f"batch_{str(i)}.png"
                fname_path = os.path.join(imgs_pth, fname)
                out_save = torch.cat((
                                    normalize(X - offset, 255).reshape(8*img_size, img_size),
                                    normalize(mask1_bin, 255).reshape(8*img_size, img_size),
                                    normalize(X_rec[:, :, 1:-1, 1:-1]  - offset, 255).reshape(8*img_size, img_size))
                                    , dim = 1).cpu().detach().numpy()
                cv2.imwrite(fname_path, out_save)

                print(f"saving batch : {i}")                
                




if __name__ == '__main__':
    infer("./inference")


