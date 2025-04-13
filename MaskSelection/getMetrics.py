import cv2
import argparse
import numpy as np
import os

def getMetrics(img1, img2):
    
    nx, ny = img1.shape
    print(nx, ny)

    mse = np.sum((img1 - img2)**2) / (nx*ny)
    mae = np.sum(np.absolute(img1 - img2)) / (nx*ny)
    psnr = 10 * np.log10( (255*255) / mse)
    
    return mse, mae, psnr

def readPGM(pth):
    img = cv2.imread(pth,flags=0)  
    return img

def calculateMetricsForDirectory(IMG_PTH1, directory_path, save_pth):
    img1 = readPGM(IMG_PTH1)
    with open(save_pth, 'w') as file:
        file.write("Filename, MSE, MAE, PSNR\n")
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                img = readPGM(file_path)
                mse, mae, psnr = getMetrics(img1, img)
                file.write(f"{filename}, {mse:.4f}, {mae:.4f}, {psnr:.4f}\n")

def main(IMG_PTH1, IMG_PTH2):
    img1 = readPGM(IMG_PTH1)
    img2 = readPGM(IMG_PTH2)

    mse, mae, psnr = getMetrics(img1, img2)

    print(f"MSE  : {mse}")
    print(f"MAE  : {mae}")
    print(f"PSNR : {psnr}")
    

if __name__ == "__main__":
    '''
    python getMetrics.py --img1_pth ch3/3.1/house/house128.pgm  --img2_pth ch3/3.1/house/house128_osm_rec.pgm 
    '''
    parser = argparse.ArgumentParser(description='Calculate metrics.')
    parser.add_argument('--img1_pth', type=str, help='Path to the PGM image 1')
    parser.add_argument('--img2_pth', type=str, help='Path to the PGM image 2')

    args = parser.parse_args()

    # PTH1 = args.img1_pth
    # PTH2 = args.img2_pth
    # PTH1 = "ch3/3.3/scarf/scarf128.pgm"
    # PTH2 = "ch3/3.3/scarf/scarf128_bh_rec.pgm"
    # main(PTH1, PTH2)

    calculateMetricsForDirectory("ch3/3.4/global/scarf/scarf128.pgm", 
                                 "ch3/3.4/global/scarf/rec",
                                 "ch3/3.4/global/scarf/metric1.txt")