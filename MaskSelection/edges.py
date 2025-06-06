import cv2
import argparse
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(".."))

from InpaintingSolver.Osmosis import OsmosisInpainting
import torch

GD_IMG_PTH = None
INIT_IMG_PTH = None

def readPGMImage( pth):
    pgm = cv2.imread(pth, cv2.IMREAD_GRAYSCALE) 
    pgm_T = torch.tensor(pgm, dtype = torch.float64)
    nx, ny = pgm_T.size()
    pgm_T = pgm_T.reshape(1, 1, nx, ny) / 255.
    # pgm_T = F.resize(pgm_T.reshape(1, 1, nx, ny) / 255., (16, 16))
    return pgm_T

def read_PGMImg(pth, blur = False):
    img = cv2.imread(pth,flags=0)  
    if blur : 
        img = cv2.GaussianBlur(img, (3,3), 0)
    return img

def sobel_edges(img, ksize = 5):
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize) # Combined X and Y Sobel Edge Detection
    max_val = np.max(sobelxy)
    return  sobelxy

def canny_edges(img, t1=100, t2=200, apertureSize=3):
    edges = cv2.Canny(image=img, threshold1=t1, threshold2=t2, apertureSize = apertureSize)
    return edges

def getDensity(img):
    nx, ny = img.shape
    x = np.count_nonzero(img == 255)
    return x / (nx*ny)

def dilate(img, k, iter ):
    kernel = np.ones((k, k), np.uint8)
    dilated_image = cv2.dilate(img, kernel, iterations=iter)    
    return dilated_image

def main():
    # read image
    img = read_PGMImg(GD_IMG_PTH, blur = True)    
    IMG_BASE_NAME = GD_IMG_PTH.split("/")[-1][:-4]   # house128
    BASE_PTH = '/'.join(GD_IMG_PTH.split("/")[:-1])  # ch3/3.2/house/

    # create canny edges and save
    print("creating canny edges")
    t1, t2 , asize = 50, 60 , 3
    c_edges = canny_edges(img, t1=t1, t2=t2, apertureSize = asize)
    den     = getDensity(c_edges)
    f_name  = IMG_BASE_NAME + "_canny_" + str(t1) + "_" + str(t2) + "_" + str(asize) + "_" + str(den) + ".pgm" # house128_canny_100_200_d.1.pgm
    cv2.imwrite(os.path.join(BASE_PTH, f_name), c_edges)

    # create sobel edges and save
    # print("creating sobel edges")
    # ksize = 5
    # s_edges = sobel_edges(img, ksize = ksize)
    # f_name  = IMG_BASE_NAME + "_sobel_" + str(ksize) + ".pgm" # house128_sobel_5.pgm
    # cv2.imwrite(os.path.join(BASE_PTH, f_name), s_edges)    

    
    # osmosis inpaint for each edge set ;
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V1 = readPGMImage(GD_IMG_PTH).to(device) + 0.001
    mask = readPGMImage(os.path.join(BASE_PTH, f_name)).to(device)
    osmosis = OsmosisInpainting(None, V1, mask, mask, offset=0.001, tau=16384, eps = 1e-9, device = device, apply_canny=True)
    osmosis.calculateWeights(False, False, False)
    osmosis.solveBatchParallel(None, None, "Stab_BiCGSTAB", 1, save_batch = [True, f"{BASE_PTH}/rec.pgm"], verbose = False)

    # save init, gd, edge mask, reconstruction 

    return



if __name__ == "__main__":

    '''
    python edges.py --gd_pth ch3/3.2/house/house128.pgm --init_pth ch3/3.2/house/house128_init.pgm

    python edges.py --gd_pth ch3/3.2/scarf/scarf128.pgm --init_pth ch3/3.2/scarf/scarf128_init.pgm

    python edges.py --gd_pth ch2/2.4/patch/masks/pepper128.pgm

    50 100 3 scarf
    
    '''

    parser = argparse.ArgumentParser(description='Edge detection using Sobel and Canny methods.')
    parser.add_argument('--gd_pth', type=str, help='Path to the guidance PGM image')
    parser.add_argument('--init_pth', type=str, help='Path to the input PGM image')
    parser.add_argument('--blur', default=False, action='store_true', help='Apply Gaussian blur to the image')
    parser.add_argument('--canny_t1', type=int, default=100, help='First threshold for the hysteresis procedure in Canny edge detection')
    parser.add_argument('--canny_t2', type=int, default=200, help='Second threshold for the hysteresis procedure in Canny edge detection')
    parser.add_argument('--sobel_ksize', type=int, default=5, help='Kernel size for Sobel operator')

    args = parser.parse_args()

    # parse arguments
    GD_IMG_PTH = args.gd_pth
    INIT_IMG_PTH = args.init_pth
    blur = args.blur
    canny_t1 = args.canny_t1
    canny_t2 = args.canny_t2
    sobel_ksize = args.sobel_ksize

    main()