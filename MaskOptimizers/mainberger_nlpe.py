# mainberger.py
import shutil
import torch
import torchvision.transforms.functional as F
import cv2
import yaml

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.abspath(".."))
from InpaintingSolver.Osmosis import OsmosisInpainting

def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None
        
def readPGMImage( pth):
    pgm = cv2.imread(pth, cv2.IMREAD_GRAYSCALE) 
    pgm_T = torch.tensor(pgm, dtype = torch.float64)
    nx, ny = pgm_T.size()
    # pgm_T = pgm_T / 255.

    # if resizing ; return pgm_T[0][0] else pgm_T
    pgm_T = F.resize(pgm_T.reshape(1, 1, nx, ny) / 255., (128, 128))
    return pgm_T[0][0]

def reconstruct(f, mask, SAVE, iter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = f.unsqueeze(0).unsqueeze(0).to(device) 
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)

    offset = 0.004
    f = f + offset

    osmosis = OsmosisInpainting(f, f, mask, mask, offset=offset, tau=16000, eps = float(config["EPS"]), device = device, apply_canny=False)
    osmosis.calculateWeights(False, False, False)
    loss, tt, max_k, df_stencils, U = osmosis.solveBatchParallel(None, None, "Stab_BiCGSTAB", 1, save_batch = [SAVE, os.path.join(exp_path, f"img_{iter}.pgm")], verbose = False)
    print(f"\n{max_k} iterations in {tt} seconds")
    return U[0,0,1:-1,1:-1]

def MSE(U, V, nxny):
    return torch.mean(torch.norm(U-V, p = 2)**2 / nxny)

def PS(f, p, q, density):
    '''
    f  : (nx, ny)
    p : fraction of mask pixels to use as candidates eg : 0.02
    q : fraction of candidates that are finally removed eg : 0.02
    return pixel mask c
    '''

    nx, ny = f.shape
    C = torch.ones(f.shape)
    K = C
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = f.to(device)
    K = K.to(device)
    iter = 0

    # all indices of mask pixels
    K_flat = K.view(-1)
    mask_indices_flat = torch.nonzero(K_flat, as_tuple=False).squeeze()
    final_selected_indices = torch.empty((0), dtype = torch.int).to(device)
    spf = len(final_selected_indices)/(nx*ny)
    
    while spf < 1 - density:
        
        # select randomly p fraction of mask pixels in K
        num_pixels = int(p * (nx*ny - len(final_selected_indices)))
        mask_indices_flat = mask_indices_flat[(mask_indices_flat == final_selected_indices.unsqueeze(-1)).sum(dim = 0) == 0]
        selected_indices_flat = mask_indices_flat[torch.randperm(mask_indices_flat.size(0))[:num_pixels]]
        K_flat[selected_indices_flat] = 0

        # copy them into T
        T_flat = K_flat.clone()
        T_flat[final_selected_indices] = 2
        T = T_flat.view(nx, ny)
        print(f"candidates pixel fraction : {len(selected_indices_flat)/(nx*ny)}")

        # compute reconstruction U given T
        if iter % config["SAVE_IMG_EVRY_ITER"] == 0:
            SAVE = True
            # temp =  torch.ones(K_flat.shape)
            # temp[final_selected_indices] = 0 
            # reconstruct(f, temp.view(nx, ny), True, 1)
        else:
            SAVE = False
        # reconstruct using only new selected candidated or in combination with final selected candidates ?? 
        U = reconstruct(f, K_flat.view(nx,ny), SAVE, iter)
        
        # compute local error u - f
        local_error = torch.abs(U - f)

        # from T select 1-q fraction of pixels with largest error and set them to one
        k = int((1-q)*num_pixels) # number of pixels to retain
        total_K = selected_indices_flat.size(0) # total candidates
        mask_local_error = torch.where((K == 0) & (T==0), local_error, torch.tensor(float('-inf')))
        mask_local_error_flat = mask_local_error.view(-1)
        topk_values, topk_indices = torch.topk(mask_local_error_flat, k=total_K, sorted=True)
        final_selected_indices = torch.cat((final_selected_indices, topk_indices[k:]), dim=0)
        K_flat[topk_indices[:k]] = 1
        spf = len(final_selected_indices)/(nx*ny)
        print(f"selected pixel fraction : {spf}", end= " ")
        
        iter += 1
    
    print("Total ITERATIONS FOR PS : ", iter)

    return mask_indices_flat

def NLPE(f, m, n, K_indices):
    '''
    K_indices : indices of mask pixels set to 1 ; |K| < density * |J|
    '''
    nx, ny = f.shape
    nxny = nx*ny
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = torch.ones((nx, ny))
    C_flat = C.view(-1)
    C_flat[K_indices] = 1
    C_new_flat  = C_flat.clone()
    iter = 0
    J_removed_indices = torch.empty((0), dtype = torch.int)

    # set device
    f = f.to(device)
    C = C.to(device)
    C_flat = C_flat.to(device)
    C_new_flat = C_new_flat.to(device)

    # reconstruct
    U = reconstruct(f, C, False, 0)

    # all mask indices J \ K
    all_indices = torch.nonzero(C_flat, as_tuple=False).squeeze()
    J_indices = torch.nonzero(C_flat, as_tuple=False).squeeze()
    J_indices = J_indices[(J_indices == K_indices.unsqueeze(-1)).sum(dim = 0) == 0]
    
    
    while iter < config["ITER"]:
        
        # randomly chose m pixels from J \ K into T
        T_indices = J_indices[torch.randperm(J_indices.size(0))[:m]]
        local_error = torch.abs(U - f)
        local_error_flat = local_error.reshape(-1)
        # local_error_T = local_error[T_indices]

        # randomly chose n pixels from K and set C_new to 1 
        K_indices_temp = K_indices[torch.randperm(K_indices.size(0))[:n]]
        C_new_flat[K_indices_temp] = 0

        # reassign based on  highest local_error_T
        not_T_indices =  all_indices[~torch.isin(all_indices, torch.tensor(T_indices))]
        local_error_flat[not_T_indices] = torch.tensor(float('-inf')).to(torch.float64)
        topk_values, T_topk_indices = torch.topk(local_error_flat, k=n)
        C_new_flat[T_topk_indices] = 1

        # U_new 
        if iter % config["SAVE_IMG_EVRY_ITER"] == 0:
            SAVE = True
        else:
            SAVE = False

        U_new = reconstruct(f, C_new_flat.view(nx, ny), True, 1)
        MSE_U_new = MSE(U_new, f, nxny)
        MSE_U     = MSE(U, f, nxny)

        if MSE_U  > MSE_U_new:
            U = U_new.clone()
            C_flat = C_new_flat.clone()
            # update K_indices ; remove K_indices_temp and add T_topk_indices
            K_indices = K_indices[(K_indices == K_indices_temp.unsqueeze(-1)).sum(dim = 0) == 0]
            K_indices = torch.cat((K_indices, T_topk_indices), dim=0)
        else:
            C_new_flat = C_flat.clone()
            # remove T_topk_indices from J_indices ?? ?? 
        
        iter += 1
        

CONFIG_YAML = 'config.yaml' 
config = read_config(CONFIG_YAML)
exp_path = os.path.join(config["PROJECT_NAME"], config["EXP_NAME"])

if __name__ == "__main__":
    
    # make exp directory and set stdout to output.txt 
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    else :
        shutil.rmtree(exp_path)
        os.makedirs(exp_path)
    # sys.stdout = open(os.path.join(exp_path, 'output.txt'),'wt')

    # copy config file to exp directory
    shutil.copyfile(CONFIG_YAML, os.path.join(exp_path, CONFIG_YAML))

    # image read
    f = readPGMImage(config['IMG_PTH'])
    print(f"Starting Probabilistic Sparsification")
    K_indices = PS(f, p = config["P"], q = config["Q"], density=config["DEN"])

    if config["NLPE"]:
        print(f"\nStarting Post Processing Non-Local Pixel Exchange")
        NLPE(f, m=config["M"], n=config["N"], K_indices=K_indices)