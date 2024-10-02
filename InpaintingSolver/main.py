from bi_cg import OsmosisInpainting
import torch
import numpy as np
import cv2

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
    return pgm_T


def normalize(X, scale = 1.):
    b, c, _ , _ = X.shape
    X = X - torch.amin(X, dim=(2,3)).view(b,c,1,1)
    X = X / (torch.amax(X, dim=(2,3)).view(b,c,1,1) + 1e-7)
    X = X * scale

    return X

if __name__ == '__main__':
    # u_pth = 'kani-init.pgm'    
    # U = readPGMImage(u_pth)

    V = readPGMImage('cameraman.pgm')
    mask = normalize(readPGMImage('cameraman-edge.pgm'))

    V1 = readPGMImage("kaniza.pgm")
    mask = normalize(readPGMImage('kaniza-edge.pgm'))
    # osmosis = OsmosisInpainting(None, V, mask, mask, offset=1, tau=300, apply_canny=False)
    # osmosis.calculateWeights(False, False, False)
    # osmosis.solve(10, save_every = 10, verbose = False)

    # V = V.repeat(4, 1, 1, 1)
    # V = torch.cat((V, V1), dim = 0)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V1 = V1.to(device)
    mask = mask.to(device)
    osmosis = OsmosisInpainting(None, V1, None, None, offset=1, tau=90000, device = device, apply_canny=False)
    osmosis.calculateWeights(False, False, False)
    osmosis.solveBatchParallel(1, save_batch = [True, "solved_b.pgm"], verbose = True)

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
