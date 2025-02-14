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

def read_PGMImg(pth, blur = False):
    img = cv2.imread(pth,flags=0)  
    if blur : 
        img = cv2.GaussianBlur(img,(3,3), SigmaX=0, SigmaY=0)
    return img

def sobel_edges(img, ksize = 5):
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize) # Combined X and Y Sobel Edge Detection
    max_val = np.max(sobelxy)
    return  sobelxy

def canny_edges(img, t1=100, t2=200):
    edges = cv2.Canny(image=img, threshold1=t1, threshold2=t2)
    return edges

def main():
    # read image
    img = read_PGMImg(GD_IMG_PTH, blur = False)    
    IMG_BASE_NAME = GD_IMG_PTH.split("/")[-1][:-4]   # house128
    BASE_PTH = '/'.join(GD_IMG_PTH.split("/")[:-1])  # ch3/3.2/house/

    # create canny edges and save
    print("creating canny edges")
    t1, t2 = 100, 150 
    c_edges = canny_edges(img, t1=t1, t2=t2)
    f_name  = IMG_BASE_NAME + "_canny_" + str(t1) + "_" + str(t2) + ".pgm" # house128_canny_100 _200.pgm
    cv2.imwrite(os.path.join(BASE_PTH, f_name), c_edges)

    # create sobel edges and save
    print("creating sobel edges")
    ksize = 5
    s_edges = sobel_edges(img, ksize = ksize)
    f_name  = IMG_BASE_NAME + "_sobel_" + str(ksize) + ".pgm" # house128_sobel_5.pgm
    cv2.imwrite(os.path.join(BASE_PTH, f_name), s_edges)    


    # osmosis inpaint for each edge set ; 
    # osmosis = OsmosisInpainting(None, X, mask1, mask2, offset=0.001, tau=tau, eps = 1e-6, device = device , apply_canny=False)
    # osmosis.calculateWeights(d_verbose=False, m_verbose=False, s_verbose=False)
    # loss3, tts, max_k, df_stencils, bicg_mat = osmosis.solveBatchParallel(df_stencils, bicg_mat, "Stab_BiCGSTAB", kmax = 1, save_batch = False, verbose = False)

    # save init, gd, edge mask, reconstruction 

    return

if __name__ == "__main__":

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