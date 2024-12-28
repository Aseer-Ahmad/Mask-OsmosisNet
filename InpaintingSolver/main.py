from Solvers import OsmosisInpainting
# from jacobi import OsmosisInpainting
import torch
import numpy as np
import cv2
import time
from torchvision import transforms, utils
import sys
import torchvision.transforms.functional as F

# setting path
sys.path.append('../')
from utils import get_dfStencil, get_bicgDict


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
    pgm_T = pgm_T.reshape(1, 1, nx, ny) / 255.
    # pgm_T = F.resize(pgm_T.reshape(1, 1, nx, ny) / 255., (128, 128))
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
    offset = 0.0001

    # V1 = readPGMImage("cameraman.pgm")
    # V1 = V1.to(device) + offset
    # mask = readPGMImage('cameraman-dmask.pgm')
    # mask = mask.to(device)

    V = readPGMImage('starfish.png')
    V = V.to(device) + offset
    
    V = V.repeat(2, 1, 1, 1)
    mask = generate_random_mask(V.shape, 0.1)
    mask = mask.to(device)

    df_stencils = get_dfStencil()
    bicg_mat = get_bicgDict()
    osmosis = OsmosisInpainting(None, V, mask, mask, offset=offset, tau=8000, eps = 1e-9, device = device, apply_canny=True)
    osmosis.calculateWeights(False, False, False)
    osmosis.solveBatchParallel(df_stencils, bicg_mat, "BiCGSTAB", 1, save_batch = [True, "solved_b.pgm"], verbose = False)


    # osmosis = OsmosisInpainting(None, V, mask, mask, offset=1, tau=300, apply_canny=False)
    # osmosis.calculateWeights(False, False, False)
    # osmosis.solve(10, save_every = 10, verbose = False)

    # image = np.array([[3,8,0],
    #                   [6,0,1],
    #                   [3,1,4]])
    # image = np.clip(image, 0, 255).astype(np.uint8)

    # with open('test1.pgm', 'wb') as f:
    #     # Write the PGM header
    #     f.write(b'P5\n')
    #     f.write(f'{image.shape[1]} {image.shape[0]}\n'.encode())
    #     f.write(b'255\n')
        
    #     # Write the image data
    #     f.write(image.tobytes())
