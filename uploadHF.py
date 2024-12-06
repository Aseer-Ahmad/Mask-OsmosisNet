from huggingface_hub import HfApi
from huggingface_hub import login
import os
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from datasets import Dataset, Features, Image, Value


# Set an environment variable
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# login()
api = HfApi()


print("starting test folder")

pth = os.path.join("dataset", "BSDS300", "images", "test")
files_list = os.listdir(pth)
# files_list = files_list[:50]
l = len(files_list)
t = 0
step_ = 1000
counter = 0


file_names = []
image_paths = []
for k, image_name in enumerate(files_list):
    image_path = os.path.join(pth, image_name)
    file_names.append(image_name)
    image_paths.append(image_path)
    print(f"\radded images {k}", end = '')

data = {
    'file_name': file_names,
    'image': image_paths
}

features = Features({
    'file_name': Value('string'),
    'image': Image()  # This tells Hugging Face to expect and display images
})

dataset = Dataset.from_dict(data, features=features)
dataset.push_to_hub("aseeransari/BSDS", split="test")




# print("starting trainfolder")

# pth = os.path.join("dataset", "imagenet", "images", "train")
# to_pth = os.path.join("dataset", "imagenet_temp", "images", "train")
# files_list = os.listdir(pth)
# l = len(files_list)
# t = 0
# step_ = 1000
# counter = 0


# file_names = []
# image_paths = []
# for k, image_name in enumerate(files_list):
#     image_path = os.path.join(pth, image_name)
#     file_names.append(image_name)
#     image_paths.append(image_path)
#     print(f"\radded images {k}", end = '')

# data = {
#     'file_name': file_names,
#     'image': image_paths
# }

# features = Features({
#     'file_name': Value('string'),
#     'image': Image()  # This tells Hugging Face to expect and display images
# })

# dataset = Dataset.from_dict(data, features=features)
# dataset.push_to_hub("aseeransari/ImageNet-Sampled", split="train", num_shards = 50)

# for i in range(0, l, step_):
#     allow_str = files_list[i:i+step_]
    
#     file_names = []
#     image_paths = []
#     for k, image_name in enumerate(allow_str):
#         image_path = os.path.join(pth, image_name)
#         file_names.append(image_name)
#         image_paths.append(image_path)
#         print(f"\radded images {k}/{step_}", end = '')

#     data = {
#         'file_name': file_names,
#         'image': image_path
#     }

#     features = Features({
#         'file_name': Value('string'),
#         'image': Image()  # This tells Hugging Face to expect and display images
#     })

#     dataset = Dataset.from_dict(data, features=features)
#     # dataset.to_parquet(os.path.join(to_pth, f"train_{counter}.parquet") )
#     dataset.push_to_hub("aseeransari/ImageNet-Sampled", num_shards = 100)
#     print(f"finished uploading : {i+step_}/{l}")
#     counter += 1

    # copy
    # for f in allow_str:
    #     shutil.copyfile(os.path.join(pth, f), os.path.join(to_pth, f))
    # print(f"finished copying : {i+step_}/{l}")

    # convert to parquet
    # image_data = []
    # for k, image_name in enumerate(allow_str):
    #     image_path = os.path.join(pth, image_name)
    #     image_bytes = Image.open(image_path).tobytes() #image_to_byte_array(image_path)
    #     image_data.append({'file_name': image_name, 'image': image_bytes})
    #     print(f"\radded images {k}/{step_}", end = '')
    # print()

    # df = pd.DataFrame(image_data)
    # df.to_parquet(os.path.join(to_pth, f"train_images_{counter}.parquet"))
    # print(f"finished parquet formatting : {i+step_}/{l}")

    # upload
    # api.upload_folder(
    #     folder_path="dataset/imagenet_temp",
    #     # path_in_repo="my-dataset/train", # Upload to a specific folder
    #     repo_id="aseeransari/ImageNet-Sampled",
    #     repo_type="dataset",
    #     # ignore_patterns="**/logs/*.txt", # Ignore all text logs
    #     # multi_commits=True,
    #     # multi_commits_verbose=True,
    #     # allow_patterns = allow_str,
    #     commit_message ="m_uploads"
    # )
    # print(f"finished uploading : {i+step_}/{l}")

    # remove parquet
    # os.remove(os.path.join(to_pth, f"train_images_{counter}.parquet"))


    # remove
    # for f in allow_str:
    #     os.remove(os.path.join(to_pth, f))
    # print(f"finished removing : {i+step_}/{l}")





# print("starting test folder")
# pth = os.path.join("dataset", "imagenet", "images", "test")
# to_pth = os.path.join("dataset", "imagenet_temp", "images", "test")
# files_list = os.listdir(pth)
# l = len(files_list)
# t = 0

# for i in range(0, l, 1000):
#     allow_str = files_list[i:i+1000]
    
#     # copy
#     for f in allow_str:
#         shutil.copyfile(os.path.join(pth, f), os.path.join(to_pth, f))
#     print(f"finished copying : {i+1000}/{l}")

#     # upload
#     api.upload_folder(
#         folder_path="dataset/imagenet_temp",
#         # path_in_repo="my-dataset/train", # Upload to a specific folder
#         repo_id="aseeransari/ImageNet-Sampled",
#         repo_type="dataset",
#         # ignore_patterns="**/logs/*.txt", # Ignore all text logs
#         multi_commits=True,
#         multi_commits_verbose=True,
#         # allow_patterns = allow_str,
#         commit_message ="m_uploads"
#     )
#     print(f"finished uploading : {i+1000}/{l}")
    

#     # remove
#     for f in allow_str:
#         os.remove(os.path.join(to_pth, f))
#     print(f"finished removing : {i+1000}/{l}")



# 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