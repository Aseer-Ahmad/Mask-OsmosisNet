from bi_cg import OsmosisInpainting
import torch
import numpy as np
import cv2


def readPGMImage( pth):
    pgm = cv2.imread(pth, cv2.IMREAD_GRAYSCALE) 
    pgm_T = torch.tensor(pgm, dtype = torch.float64)
    nx, ny = pgm_T.size()
    pgm_T = pgm_T.reshape(1, 1, nx, ny)
    return pgm_T

if __name__ == '__main__':
    u_pth = 'InpaintingSolver/svalbard-init.pgm'
    v_pth = 'InpaintingSolver/svalbard.pgm'
    
    U = readPGMImage(u_pth)
    V = readPGMImage(v_pth)
    osmosis = OsmosisInpainting(U, V, None, 10, 1)
    osmosis.calculateWeights()
    osmosis.solve()


    # pth = 'InpaintingSolver/test.pgm'
    # V = osmosis.readPGMImage(pth)
    # osmosis.V = V + 1
    # osmosis.getDriftVectors(True)
    # osmosis.getStencilMatrices(1,True)
    # pth = 'InpaintingSolver/test1.pgm'
    # U = osmosis.readPGMImage(pth)
    # U = U + 1
    # x = osmosis.applyStencil(U, True)


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
