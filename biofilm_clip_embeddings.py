import time
start = time.time()
import cv2
import os
import platform
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import torch
import clip
from PIL import Image
import numpy as np
from natsort import natsorted

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Loading model took {time.time() - start} seconds")

home_dir = "."
if platform.system() == "Darwin":
    home_dir = "."
else:
    home_dir = "/mnt/e/ImageLibrary"

dir = f"."
paths = os.listdir(dir)
folders = []
for entry in paths: 
    directory = f"./{entry}"
    if os.path.isdir(directory):
        if "Drawer" in directory:
            folders.append(f"{entry}")

images_dirs = []
for folder in folders:
    for sub_folder in os.listdir(folder):
        if os.path.isdir(f"./{folder}/{sub_folder}/results_images"):
            images_dirs.append(f"./{folder}/{sub_folder}/results_images")
for dir in images_dirs:
    print(dir)

labels_dict = {}
labels = pd.read_csv("ReplicatePositions.csv")
for _, row in labels.iterrows():
    labels_dict[row.iloc[0]] = row.iloc[1]
print(labels_dict)


classes = np.unique(labels.iloc[:,1])
print(classes)
print(home_dir)
os.makedirs(f"{home_dir}/Embeddings", exist_ok=True)

for c in classes:
    os.makedirs(f"{home_dir}/Embeddings/{c}", exist_ok=True)


plate_index = 1
for dir in images_dirs:
    path = f"{dir}"
    for file in natsorted(os.listdir(path)):
        if file.find("mask") == -1 and file.find("Thumb") == -1:
            magnification = ""
            if file.find("4x") > 0:
                magnification = "4x"
            elif file.find("10x") > 0:
                magnification = "10x"
            elif file.find("20x") > 0:
                magnification = "20x"
            elif file.find("40x") > 0:
                magnification = "40x"

            well = file[:3]
            if well[-1] == '_':
                well = well[:2]
            print(file)
            embeddings_dir = f"{home_dir}/Embeddings/{labels_dict[well]}"
            print("MAGNIFICATION: ", magnification)
            if not os.path.exists(f"{embeddings_dir}/{magnification}"):
                os.makedirs(f"{embeddings_dir}/{magnification}")
            embeddings_dir = f"{home_dir}/Embeddings/{labels_dict[well]}/{magnification}"
        
            image_stack = []
            file_path = f"{path}/{file}"
            print(f"Storing image stack and computing embeddings for {file_path}, strain {labels_dict[well]}")
            ret,images = cv2.imreadmulti(mats=image_stack,
                                         filename=file_path,
                                         start=0,
                                         count=31,
                                         flags=cv2.IMREAD_ANYCOLOR)
            embeddings = []
            embeddings_start = time.time()
            with torch.no_grad():
                for i in range(0, len(images)):
                    processed = preprocess(Image.fromarray(images[i])).unsqueeze(0).to(device)
                    image_features = model.encode_image(processed).cpu().numpy()[0]
                    embeddings.append(image_features)
            embeddings_end = time.time()
            embeddings = np.array(embeddings)
            print(f"{embeddings_dir}/{file[:len(file) - 4]}.npy")
            np.save(f"{embeddings_dir}/plate_{plate_index}_{file[:len(file) - 4]}.npy", embeddings)
            
            print(f"Generating embeddings for {file_path} took {embeddings_end - embeddings_start} seconds")
            print()
    plate_index += 1