import cv2
import numpy as np

def resizePGMImage( pth):
    pgm = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)     
    resized_image = cv2.resize(pgm, (128, 128))
    cv2.imwrite("./natural/pepper128.pgm", resized_image)

def generate_init(pth) : 
    image = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)     
    avg_value = image.mean(axis=(0, 1))  
    new_image = np.full_like(image, avg_value, dtype=np.uint8)
    cv2.imwrite("./natural/pepper128_init.pgm", new_image)

resizePGMImage("./natural/pepper.pgm")
generate_init("./natural/pepper128.pgm")