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

if __name__ == '__main__':
    # u_pth = 'kani-init.pgm'
    v_pth = 'cameraman.pgm'
    mask_pth  = 'cameraman-edge.pgm'
    
    # U = readPGMImage(u_pth)
    V = readPGMImage(v_pth)
    V1 = readPGMImage("scarf.pgm")
    mask = readPGMImage(mask_pth)

    # osmosis = OsmosisInpainting(None, V1, None, None, offset=1, tau=10, apply_canny=False)
    # osmosis.calculateWeights(False, False, False)
    # osmosis.solve(20, save_every = 20, verbose = True)

    # V = V.repeat(4, 1, 1, 1)
    V = torch.cat((V, V1), dim = 0)

    osmosis = OsmosisInpainting(None, V, None, None, offset=1, tau=10, device = None, apply_canny=False)
    osmosis.calculateWeights(False, False, False)
    osmosis.solveBatchProcess(2, save_batch = True, verbose = True)

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
