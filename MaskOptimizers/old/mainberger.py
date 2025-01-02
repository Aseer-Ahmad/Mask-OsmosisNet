# mainberger.py
import torch
import torchvision.transforms.functional as F
import cv2

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.abspath(".."))
from InpaintingSolver.Solvers import OsmosisInpainting

# make exp directory and write 
exp_path = os.path.join("mainberger_exp", "EXP_4")
if not os.path.isdir(exp_path):
    os.makedirs(exp_path)
sys.stdout = open(os.path.join(exp_path, 'output.txt'),'wt')

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

    osmosis = OsmosisInpainting(f, f, mask, mask, offset=offset, tau=16000, eps = 1e-9, device = device, apply_canny=False)
    osmosis.calculateWeights(False, False, False)
    loss, tt, max_k, df_stencils, U = osmosis.solveBatchParallel(None, None, "Stab_BiCGSTAB", 1, save_batch = [SAVE, os.path.join(exp_path, f"img_{iter}.pgm")], verbose = False)
    print(f" {max_k} iterations in {tt} seconds")
    return U[0,0,1:-1,1:-1]

# method can be trapped in local minima
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
        K = K_flat.view(nx, ny)

        # copy them into T
        T_flat = K_flat.clone()
        T_flat[final_selected_indices] = 2
        T = T_flat.view(nx, ny)
        print(f"candidates pixel fraction : {len(selected_indices_flat)/(nx*ny)}")

        # compute reconstruction U given T
        if iter % 50 == 0:
            SAVE = True
        else:
            SAVE = False
        U = reconstruct(f, K, SAVE, iter)

        # compute local error u - f
        local_error = torch.abs(U - f)

        # from T select 1-q fraction of pixels with largest error and set them to one
        k = int((1-q)*num_pixels) # number of pixels to retain
        total_K = selected_indices_flat.size(0) # total candidates
        mask_local_error = torch.where((K == 0) & (T==0), local_error, torch.tensor(float('-inf')))
        mask_local_error_flat = mask_local_error.view(-1)
        K_flat = K.view(-1)
        topk_values, topk_indices = torch.topk(mask_local_error_flat, k=total_K, sorted=True)
        final_selected_indices = torch.cat((final_selected_indices, topk_indices[k:]), dim=0)
        K_flat[topk_indices[:k]] = 1
        spf = len(final_selected_indices)/(nx*ny)
        print(f"selected pixel fraction : {spf}", end= " ")
        
        iter += 1

# imgs/natural/scarf.pgm
if __name__ == "__main__":
    f = readPGMImage('../imgs/natural/scarf.pgm')

    PS(f, p = 0.02, q = 0.02, density=0.3)