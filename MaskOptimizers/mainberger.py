# mainberger.py
import torch
import cv2
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(".."))
from InpaintingSolver.Solvers import OsmosisInpainting
import torchvision.transforms.functional as F

def readPGMImage( pth):
    pgm = cv2.imread(pth, cv2.IMREAD_GRAYSCALE) 
    pgm_T = torch.tensor(pgm, dtype = torch.float64)
    nx, ny = pgm_T.size()
    # pgm_T = pgm_T / 255.

    # when resizing return pgm_T[0][0]
    pgm_T = F.resize(pgm_T.reshape(1, 1, nx, ny) / 255., (128, 128))
    return pgm_T[0][0]

def reconstruct(f, mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = f.unsqueeze(0).unsqueeze(0).to(device) 
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)

    offset = 0.004
    f = f + offset

    osmosis = OsmosisInpainting(None, f, mask, mask, offset=offset, tau=16000, eps = 1e-3, device = device, apply_canny=False)
    osmosis.calculateWeights(False, False, False)
    loss, tt, max_k, df_stencils, U = osmosis.solveBatchParallel(None, None, "Stab_BiCGSTAB", 1, save_batch = [True, "test.pgm"], verbose = False)
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

    while iter < 10000:
        
        # select randomly p fraction of mask pixels in K
        num_pixels = int(p * torch.sum(K).item())
        K_flat = K.view(-1)
        mask_indices_flat = torch.nonzero(K_flat, as_tuple=False).squeeze()
        selected_indices_flat = mask_indices_flat[torch.randperm(mask_indices_flat.size(0))[:num_pixels]]
        K_flat[selected_indices_flat] = 0
        K = K_flat.view(nx, ny)

        # copy them into T
        print(f"candidates pixel fraction : {1 - torch.norm(K, p=1)/(nx*ny)}")

        # compute reconstruction U given T
        U = reconstruct(f, K)

        # compute local error u - f
        local_error = torch.abs(U - f)

        # from T select 1-q fraction of pixels with largest error and set them to one
        k = int((1-q)*(p * nx * ny))
        mask_local_error = torch.where(K == 0, local_error, torch.tensor(float('-inf')))
        mask_local_error_flat = mask_local_error.view(-1)
        K_flat = K.view(-1)
        _, topk_indices = torch.topk(mask_local_error_flat, k=k)
        K_flat[topk_indices] = 1
        K = K_flat.view(nx, ny)
        print(f"selected pixel fraction : {1 - torch.norm(K, p=1)/(nx*ny)}")
        
        iter += 1

# imgs/natural/scarf.pgm
if __name__ == "__main__":
    f = readPGMImage('../imgs/natural/scarf.pgm')

    PS(f, 0.02, 0.02, 0.3)