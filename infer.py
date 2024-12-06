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
import pandas as pd
from torchvision.utils import save_image

import cv2

from torchmetrics.image import PeakSignalNoiseRatio

class Binarize(object):
    @staticmethod
    def hard_rounding(x):
        return torch.floor(x + 0.6)

def calculate_metrics(input_tensor, target_tensor, max_pixel_value=255.0):
    mse_per_batch = torch.mean((input_tensor - target_tensor) ** 2, dim=(1, 2, 3))
    psnr_per_batch = 10 * torch.log10((max_pixel_value ** 2) / (mse_per_batch + 1e-8))
    return mse_per_batch.tolist(), psnr_per_batch.tolist()

def save_imgs(X, save_dir, names, suffix):
    for i in range(X.size(0)):
        file_path = os.path.join(save_dir, names[i].split(".")[0] + f"_{suffix}.png")
        cv2.imwrite(file_path, X[i][0].cpu().detach().numpy())
        # save_image(X[i], file_path)

# def normalize(X, min_val = 0, max_val = 1):
#     b, c, _ , _ = X.shape
#     X = X - torch.amin(X, dim=(2,3)).view(b,c,1,1)
#     X = X / (torch.amax(X, dim=(2,3)).view(b,c,1,1) + 1e-7)
#     X = X * scale
#     return X

def get_mask_density(mask):
    return (torch.norm(mask, p = 1, dim = (1, 2, 3)) / (mask.shape[2]*mask.shape[3])).tolist()

def get_model_by_task(task):
    if task == 'unet_single' :
        return UNet(1, 1, tar_den = 0.1)
    elif task == 'unet_double' :
        return UNet(1, 2, tar_den = 0.1)
    elif task == 'masknet_single' :
        return MaskNet(1, 1, tar_den = 0.1)

def infer(infer_path):

    infer_config = read_config(os.path.join(infer_path, "infer.yaml"))
    print(infer_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}\n")

    psnr  = PeakSignalNoiseRatio().to(device)
    mse   = torch.nn.MSELoss()

    for key, task  in infer_config["TASK"].items():
        print(f"inferring for task : {key}")

        # init
        img_size = task["IMG_SIZE"]
        offset   = 1
        img_names, mse_c_list, psnr_c_list, mse_nb_list, psnr_nb_list, mse_b_list, psnr_b_list  = [],[],[],[],[],[],[] 
        canny_mask_den_list, bin_mask_den_list, batch_name_list = [], [], []

        # create directories
        f_name = task['MODEL_CKP_PTH'].split(".")[0].split("/")[-1] \
               + "_" + task["TYPE"] + "_" + task["BIN_METHOD"] 
        
        fld_pth  = os.path.join(infer_path, f_name)
        imgs_pth = os.path.join(fld_pth, "images")
        imgs_sep_pth = os.path.join(fld_pth, "images separate")
        
        # if path exist -> skip
        if os.path.exists(imgs_pth):
            print(f"skipping task : {key} , PATH ALREADY EXIST")
            continue
        else:
            os.makedirs(imgs_pth)
            os.makedirs(imgs_sep_pth)

        # load models
        if task['MODEL_CKP_PTH'] :
            model    = get_model_by_task(task['TYPE'])
            model, _ = loadCheckpoint(model, None, task['MODEL_CKP_PTH'])
            model    = model.double()
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

        with torch.no_grad():
            for i, (X, X_names) in enumerate(test_dataloader): 

                X = X.to(device, dtype=torch.float64) # [0,1]
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
                osmosis = OsmosisInpainting(None, X, None, None, offset=offset, tau=30000, eps = 1e-11, device = device, apply_canny=True)
                osmosis.calculateWeights(d_verbose=False, m_verbose=False, s_verbose=False)
                save_batch = [False]
                loss3, tts, max_k, df_stencils, X_rec_c = osmosis.solveBatchParallel(None, None, 1, save_batch = save_batch, verbose = False)
                mask1_c = osmosis.mask1.double()
                

                # solve for non-binary masks
                osmosis = OsmosisInpainting(None, X, mask1, mask2, offset=offset, tau=30000, eps = 1e-11, device = device, apply_canny=False)
                osmosis.calculateWeights(d_verbose=False, m_verbose=False, s_verbose=False)
                save_batch = [False]
                loss3, tts, max_k, df_stencils, X_rec_nb = osmosis.solveBatchParallel(None, None, 1, save_batch = save_batch, verbose = False)


                # solve for binary masks
                osmosis = OsmosisInpainting(None, X, mask1_bin, mask2_bin, offset=offset, tau=30000, eps = 1e-11, device = device, apply_canny=False)
                osmosis.calculateWeights(d_verbose=False, m_verbose=False, s_verbose=False)
                save_batch = [False]
                loss3, tts, max_k, df_stencils, X_rec = osmosis.solveBatchParallel(None, None, 1, save_batch = save_batch, verbose = False)


                # save results and csv
                X_norm = X * 255 # normalize(X , 255)
                mask1_c_norm = mask1_c[:, :, 1:-1, 1:-1] * 255.
                X_rec_c_norm = (X_rec_c[:, :, 1:-1, 1:-1]-offset) * 255 # normalize(X_rec_c[:, :, 1:-1, 1:-1]-offset, 255)

                mask1_norm = mask1 * 255.
                mask2_norm = mask2 * 255.
                X_rec_nb_norm = (X_rec_nb[:, :, 1:-1, 1:-1]-offset) * 255

                mask1_bin_norm = mask1_bin * 255.
                mask2_bin_norm = mask2_bin * 255.
                X_rec_b_norm  = (X_rec[:, :, 1:-1, 1:-1]-offset)*  255

                mse_c, psnr_c = calculate_metrics(X_norm, X_rec_c_norm)
                mse_nb, psnr_nb = calculate_metrics(X_norm, X_rec_nb_norm)
                mse_b, psnr_b = calculate_metrics(X_norm, X_rec_b_norm)

                fname = f"batch_{str(i)}.png"
                fname_path = os.path.join(imgs_pth, fname)
                if mask.shape[1] == 2:
                    out_save = torch.cat((
                                        X_norm.mT.reshape(8*img_size, img_size),
                                        mask1_c_norm.reshape(8*img_size, img_size),
                                        X_rec_c_norm.reshape(8*img_size, img_size),
                                        mask1_norm.mT.reshape(8*img_size, img_size),
                                        mask2_norm.mT.reshape(8*img_size, img_size),
                                        X_rec_nb_norm.reshape(8*img_size, img_size),
                                        mask1_bin_norm.mT.reshape(8*img_size, img_size),
                                        mask2_bin_norm.mT.reshape(8*img_size, img_size),
                                        X_rec_b_norm.reshape(8*img_size, img_size) )
                                        , dim = 1).cpu().detach().numpy().T
                else:
                    out_save = torch.cat((
                                        X_norm.mT.reshape(8*img_size, img_size),
                                        mask1_c_norm.reshape(8*img_size, img_size),
                                        X_rec_c_norm.reshape(8*img_size, img_size),
                                        mask1_norm.mT.reshape(8*img_size, img_size),
                                        X_rec_nb_norm.reshape(8*img_size, img_size),
                                        mask1_bin_norm.mT.reshape(8*img_size, img_size),
                                        X_rec_b_norm.reshape(8*img_size, img_size) )
                                        , dim = 1).cpu().detach().numpy().T
                cv2.imwrite(fname_path, out_save)
                print(f"saving batch : {i}")   

                # save separate  
                save_imgs(X_norm, imgs_sep_pth, X_names, "")
                save_imgs(X_rec_b_norm.mT, imgs_sep_pth, X_names, "rec")
                save_imgs(mask1_bin_norm, imgs_sep_pth, X_names, "mask1_bin")
                save_imgs(mask2_bin_norm, imgs_sep_pth, X_names, "mask2_bin")
                

                batch_name_list.extend([f"batch_{str(i)}"]*8)
                img_names.extend(X_names)
                canny_mask_den_list.extend(get_mask_density(mask1_c))
                bin_mask_den_list.extend(get_mask_density(mask1_bin))
                mse_c_list.extend(mse_c)
                psnr_c_list.extend(psnr_c)
                mse_nb_list.extend(mse_nb)
                psnr_nb_list.extend(psnr_nb)
                mse_b_list.extend(mse_b)
                psnr_b_list.extend(psnr_b) 

                csv_dict = {
                    "batch no." : batch_name_list,
                    "image name" : img_names,
                    "canny mask den" : canny_mask_den_list,
                    "bin mask den" : bin_mask_den_list,
                    "MSE canny" : mse_c_list,
                    "PSNR canny" : psnr_c_list,
                    "MSE non bin" : mse_nb_list,
                    "PSNR non bin" : psnr_nb_list,
                    "MSE bin" : mse_b_list,
                    "PSNR bin" : psnr_b_list 
                }

                df = pd.DataFrame(csv_dict)
                df.to_csv( os.path.join(fld_pth, "data.csv"), sep=',', encoding='utf-8', index=False, header=True)

                


if __name__ == '__main__':
    infer("./inference")


