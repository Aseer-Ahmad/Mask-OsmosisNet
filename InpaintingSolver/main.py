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
    u_pth = 'InpaintingSolver/sc-init.pgm'
    v_pth = 'InpaintingSolver/sc.pgm'
    mask_pth  = 'InpaintingSolver/cameraman-edge.pgm'
    
    U = readPGMImage(u_pth)
    V = readPGMImage(v_pth)
    mask = readPGMImage(mask_pth)

    osmosis = OsmosisInpainting(U, V, mask, mask, offset=1, tau=100)
    # osmosis.writePGMImage(osmosis.U[0][0].T.numpy().T, "cm-init.pgm")
    osmosis.calculateWeights(False, False)
    osmosis.solve(1, False)


    # pth = 'InpaintingSolver/test.pgm'
    # V = osmosis.readPGMImage(pth)
    # osmosis.V = V + 1
    # osmosis.getDriftVectors(True)
    # osmosis.getStencilMatrices(1,True)
    # pth = 'InpaintingSolver/test1.pgm'
    # U = osmosis.readPGMImage(pth)
    # U = U + 1
    # x = osmosis.applyStencil(U, True)

    # Um = U[0][0][190:200, 215:230]
    # write_tensor_to_pgm('sc-init.pgm', Um)
    # Vm = V[0][0][190:200, 215:230]
    # write_tensor_to_pgm('sc.pgm', Vm)

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
