#!/usr/bin/env python
import os
import shutil
import sys

# code to use after extracting <imagenet_data> inside the dataset folder

train_pth = os.path.join("dataset", "imagenet", "images", "train")
test_pth  = os.path.join("dataset", "imagenet", "images", "test")

if not os.path.isdir(train_pth):
    os.makedirs(train_pth)
    os.makedirs(test_pth)

old_train_pth = os.path.join("dataset", "mydataset", "my_train_set", "images")
old_test_pth = os.path.join("dataset", "mydataset", "my_val_set", "images")

print(f"moving from {old_train_pth} to {train_pth}")
shutil.move(old_train_pth, train_pth) 

print(f"moving from {old_test_pth} to {test_pth}")
shutil.move(old_test_pth, test_pth) 


# imagenet
data_pth = "dataset\imagenet\images"

train_pth = os.path.join(data_pth, "train")
test_pth = os.path.join(data_pth, "test")

train_files = os.listdir(train_pth)
with open(os.path.join(data_pth, 'iids_train.txt'), 'w') as f:
    for line in train_files:
        f.write(f"{line}\n")
print(f"written {len(train_files)} files")


test_files = os.listdir(test_pth)
with open(os.path.join(data_pth, 'iids_test.txt'), 'w') as f:
    for line in test_files:
        f.write(f"{line}\n")
print(f"written {len(test_files)} files")
