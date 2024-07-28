from bi_cg import OsmosisInpainting
import torch
import numpy as np

if __name__ == '__main__':
    osmosis = OsmosisInpainting()
    pth = 'InpaintingSolver/test.pgm'
    V = osmosis.readPGMImage(pth)
    osmosis.V = V + 1
    osmosis.getDriftVectors()
    osmosis.getStencilMatrices(1)
    pth = 'InpaintingSolver/test1.pgm'
    U = osmosis.readPGMImage(pth)
    U = U + 1
    x = osmosis.applyStencil(U, True)

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
