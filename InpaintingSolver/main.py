from bi_cg import OsmosisInpainting
import torch

if __name__ == '__main__':
    osmosis = OsmosisInpainting()
    pth = 'InpaintingSolver/svalbard.pgm'
    img = osmosis.readPGMImage(pth)
    osmosis.V = img
    osmosis.getDriftVectors(verbose=True)
    print(osmosis.d1)
    print(osmosis.d2)
