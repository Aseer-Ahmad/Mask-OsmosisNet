import os
import sys
import cv2
import glob
import shutil

DIR_PTH = 'ch3/3.3'

# copy paste same path and files
DIR_PNG_PTH = DIR_PTH + "_PNG"
shutil.copytree(DIR_PTH, DIR_PNG_PTH,dirs_exist_ok=True)

# get all .pgm files
pgm_file_list = glob.glob(DIR_PNG_PTH + "/**/*.pgm", recursive=True)
print(f"total PGM files in directory : {len(pgm_file_list)}")

for pgm_file in pgm_file_list:
    base_filname = pgm_file[:-3]
    png_filename = base_filname + "png"

    img = cv2.imread(pgm_file,flags=0)  
    
    cv2.imwrite(png_filename, img)
    print(f"written : {png_filename}")

    # remove pgm
    os.remove(pgm_file)

