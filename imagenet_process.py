import os
import shutil
import sys

data_pth = "dataset\BSDS300\images"

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
