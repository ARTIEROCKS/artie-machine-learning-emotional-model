import yaml
import sys
import imutils
import cv2
import os
from tqdm import tqdm
from pathlib import Path

# Loading the parameters
params_file = sys.argv[1]
with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

dataset_emotions_augmented_path = params['data_augmentation']['dataset_emotions_augmented_path']
dataset_images_augmented_path = params['data_augmentation']['dataset_images_augmented_path']
normalization_path = params['normalization']['normalization_path']

Path(normalization_path).mkdir(parents=True, exist_ok=True)

# Getting the list of images
image_list = list(Path(dataset_images_augmented_path).rglob("*.png"))
image_list_size = len(image_list)

print("Image processing is starting.")

# loop over the input images
for inputPath in tqdm(image_list, desc="Procesando", unit="imagen"):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(str(inputPath))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # write the output image to disk
    resized = imutils.resize(gray, width=48, height=48)
    cv2.imwrite(os.path.join(normalization_path, str(inputPath).split("/")[-1]), resized)

    # display the output images
    # cv2.imshow("Resized", resized)
    # cv2.waitKey(1)