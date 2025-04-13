
# from jacobi import OsmosisInpainting
import torch
import numpy as np
import cv2
import time
from torchvision import transforms, utils
import sys
import torchvision.transforms.functional as F
import torch
import random

# torch.manual_seed(1)
# setting path
sys.path.append('../')

# from utils import get_dfStencil, get_bicgDict
from InpaintingSolver.Osmosis import OsmosisInpainting
from InpaintingSolver.Diffusion import DiffusionInpainting


def write_tensor_to_pgm(filename, tensor):
    tensor = torch.clamp(tensor, 0, 255).byte()
    image = tensor.numpy()
    height, width = image.shape

    with open(filename, 'wb') as f:
        f.write(b'P5\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(b'255\n')
        f.write(image.tobytes())

def readPGMImage( pth):
    pgm = cv2.imread(pth, cv2.IMREAD_GRAYSCALE) 
    pgm_T = torch.tensor(pgm, dtype = torch.float64)
    nx, ny = pgm_T.size()
    # Perform center square crop
    min_dim = min(nx, ny)
    top = (nx - min_dim) // 2
    left = (ny - min_dim) // 2
    pgm_T = pgm_T[top:top + min_dim, left:left + min_dim]
    pgm_T = F.resize(pgm_T.reshape(1, 1, min_dim, min_dim) / 255., (256, 256)) 

    # pgm_T = F.resize(pgm_T.reshape(1, 1, nx, ny) / 255., (128, 128))
    return pgm_T

def readPGMBinaryImage( pth):
    pgm = cv2.imread(pth, cv2.IMREAD_GRAYSCALE) 
    pgm_T = torch.tensor(pgm, dtype = torch.float64)
    nx, ny = pgm_T.size()
    pgm_T = pgm_T.reshape(1, 1, nx, ny) 
    return pgm_T


def normalize(X, scale = 1.):
    b, c, _ , _ = X.shape
    X = X - torch.amin(X, dim=(2,3)).view(b,c,1,1)
    X = X / (torch.amax(X, dim=(2,3)).view(b,c,1,1) + 1e-7)
    X = X * scale

    return X

def generate_random_mask(shape, density):
    mask = torch.rand(shape)
    binary_mask = (mask < density).int()
    return binary_mask


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offset = 0.001

    V1 = readPGMImage("ch2/2.4/senstivity/lisa2.pgm").to(device) + offset
    # V1_init = readPGMImage("ch2/2.4/patch/pepper128_init.pgm").to(device) + offset
    mask = readPGMImage("ch2/2.4/senstivity/fry2.pgm").to(device)
    # print(mask)
    nx, ny = V1.shape[2],V1.shape[3]

    # V = readPGMImage('scarf.pgm')
    # V = V.to(device) + offset
    
    # V = V.repeat(1, 1, 1, 1)
    # mask = generate_random_mask(V.shape, 0.1)
    # mask = mask.to(device)

    # df_stencils = get_dfStencil()
    # bicg_mat = get_bicgDict()
    
    # Osmosis
    # for i in range(1,15):
    #     mask = readPGMImage("ch2/2.4/patch/pepper128_canny_170_180_0.10.pgm").to(device)
    #     kernel_size = 50
    #     top = random.randint(0, nx - kernel_size)
    #     left = random.randint(0, ny - kernel_size)
    #     mask[0, 0, top:top+kernel_size, left:left+kernel_size] = 0
        
    osmosis = OsmosisInpainting(None, V1, mask, mask, offset=offset, tau=65536, eps = 1e-9, device = device, apply_canny=True)
    osmosis.calculateWeights(False, False, False)
    osmosis.solveBatchParallel(None, None, "Stab_BiCGSTAB", 1, save_batch = [True, f"ch2/2.4/senstivity/lisa256_rec.pgm"], verbose = False)
    
    init = osmosis.init[0][0].cpu().detach().numpy().T * 255
    d1 = osmosis.d1[0][0].cpu().detach().numpy().T * 255
    d2 = osmosis.d2[0][0].cpu().detach().numpy().T * 255
    print(d1.shape, d2.shape)

    # cv2.imwrite("ch2/2.4/patch/pepper128_init.pgm", init)
    cv2.imwrite("ch2/2.4/senstivity/lisa256_mask.pgm", osmosis.mask1[0][0].cpu().detach().numpy().T*255. )
    cv2.imwrite("ch2/2.4/senstivity/lisa256.pgm", V1[0][0].cpu().detach().numpy() * 255.)
    cv2.imwrite("ch2/2.4/senstivity/lisa256_d1.pgm", d1)
    cv2.imwrite("ch2/2.4/senstivity/lisa256_d2.pgm", d2)
    


    
    # Diffusion
    # U    = readPGMImage("klein.pgm").to(device)
    # mask = generate_random_mask(U.shape, 0.1).to(device)
    # mask = readPGMImage("klein-mask.pgm").to(device)
    # diffusion = DiffusionInpainting(U, mask , tau=16000, eps = 1e-5, device = device, apply_canny=False)
    # diffusion.prepareInp()
    # loss, tt, max_k, df_stencils, U = diffusion.solveBatchParallel(df_stencils, bicg_mat, "CG", 1,  save_batch = [True, "solved_d.pgm"], verbose = False)
    # print(loss, max_k)

