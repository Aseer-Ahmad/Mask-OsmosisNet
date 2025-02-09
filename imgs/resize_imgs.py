import cv2
import numpy as np

def resizePGMImage( pth):
    pgm = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)     
    resized_image = cv2.resize(pgm, (128, 128))
    cv2.imwrite("scarf_s.pgm", resized_image)

def generate_init(pth) : 
    image = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)     
    avg_value = image.mean(axis=(0, 1))  # Compute mean across width and height
    new_image = np.full_like(image, avg_value, dtype=np.uint8)
    cv2.imwrite("scarf_s_init.pgm", new_image)

resizePGMImage("./natural/scarf_s.pgm")
generate_init("scarf_s.pgm")