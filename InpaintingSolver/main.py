from bi_cg import OsmosisInpainting
import torch
import numpy as np

if __name__ == '__main__':
    osmosis = OsmosisInpainting()
    pth = 'InpaintingSolver/test.pgm'
    img = osmosis.readPGMImage(pth)
    osmosis.V = img + 1
    osmosis.getDriftVectors(verbose=True)


    # image = np.array([[3,5,7],
    #                   [5,2,9],
    #                   [1,8,2]])
    # image = np.clip(image, 0, 255).astype(np.uint8)

    # with open('test.pgm', 'wb') as f:
    #     # Write the PGM header
    #     f.write(b'P5\n')
    #     f.write(f'{image.shape[1]} {image.shape[0]}\n'.encode())
    #     f.write(b'255\n')
        
    #     # Write the image data
    #     f.write(image.tobytes())
