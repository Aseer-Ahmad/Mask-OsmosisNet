from bi_cg import OsmosisInpainting
import torch
import numpy as np
import cv2
import time
from torchvision import transforms, utils

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
    pgm_T = pgm_T.reshape(1, 1, nx, ny)
    # norm = transforms.Normalize(mean = [0.], std = [1.])
    # pgm_T = norm(pgm_T)
    pgm_T = normalize(pgm_T)
    return pgm_T


def normalize(X, scale = 1.):
    b, c, _ , _ = X.shape
    X = X - torch.amin(X, dim=(2,3)).view(b,c,1,1)
    X = X / (torch.amax(X, dim=(2,3)).view(b,c,1,1) + 1e-7)
    X = X * scale

    return X

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    V = readPGMImage('cameraman.pgm')
    V = V.to(device)
    mask = normalize(readPGMImage('cameraman-edge.pgm'))

    V1 = readPGMImage("scarf.pgm")
    V1 = V1.to(device)
    mask = normalize(readPGMImage('kaniza-edge.pgm'))
    mask = mask.to(device)

    
    V1 = V1.repeat(16, 1, 1, 1)
    mask = mask.repeat(16, 1, 1, 1)
    # V = V.to(device)
    print(V1)

    osmosis = OsmosisInpainting(None, V1, None, None, offset=1, tau=9000, device = device, apply_canny=True)
    st = time.time()
    osmosis.calculateWeights(False, False, False)
    et = time.time()
    print(f"calculate weights total time : {(et - st)} sec")
    osmosis.solveBatchParallel(1, save_batch = [True, "solved_b.pgm"], verbose = False)


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
