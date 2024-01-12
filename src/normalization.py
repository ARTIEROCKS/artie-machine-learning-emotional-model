import yaml
import sys
import imutils
import cv2
import os
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path


# Function to normalize emotional state for CK+
def ck_emotional_state_normalization(emotional_state_path, augmented_images_path, root_path, csv_path):
    print("Image List and Emotion Collection")

    emotions_list = []
    corresponding_images = []
    emotions_file_list = list(Path(emotional_state_path).rglob("*.txt"))

    for emotion_file in tqdm(emotions_file_list, desc="Processing CK+ emotions", unit="image"):
        # Reading the emotional state from the emotions_file_list
        f = open(str(emotion_file), "r")
        contents = f.read()
        value = float(contents)
        emotion = int(value)

        # Gets the name of the original corresponding image
        emotion_file_splitted = str(emotion_file).split("/")
        emotion_file_name = emotion_file_splitted[len(emotion_file_splitted) - 1]
        emotion_file_name_splitted = emotion_file_name.split(".")
        emotion_file_name_splitted = emotion_file_name_splitted[0].split("_emotion")
        emotion_file_name = emotion_file_name_splitted[0]

        # Gets the augmented files corresponding to the emotional state
        emotion_image_list = list(Path(root_path).rglob(emotion_file_name + "*.png"))
        for image in emotion_image_list:
            emotions_list.append(emotion)
            corresponding_images.append(str(image))

    # Adds all the rows and columns to the emotion data frame
    d = {'emotion': emotions_list, 'corresponding_image': corresponding_images}
    emotions_df = pd.DataFrame(data=d)
    emotions_df.to_csv(csv_path, index=False, header=True, mode='a')


# Function to normalize the CK+ augmented files
def ck_normalization(ds_images_augmented_path):
    # Getting the list of images
    image_list = list(Path(ds_images_augmented_path).rglob("*.png"))

    print("CK+ Image processing is starting.")

    # loop over the input images
    for inputPath in tqdm(image_list, desc="Processing CK+", unit="image"):
        # load the image, convert it to grayscale, and describe it
        image = cv2.imread(str(inputPath))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # write the output image to disk
        resized = imutils.resize(gray, width=48, height=48)
        cv2.imwrite(os.path.join(normalization_path, str(inputPath).split("/")[-1]), resized)

        # display the output images
        # cv2.imshow("Resized", resized)
        # cv2.waitKey(1)


# Function to normalize FER 2013 dataset
def fer2013_normalization(dataset_path, destination_path, emotion_csv_file):
    print("FER 2013 Image processing is starting.")

    # Getting training images
    emotional_states = [name for name in os.listdir(dataset_path) if
                        os.path.isdir(os.path.join(dataset_path, name))]
    # Create a dictionary
    mapping = {
        'neutral': 0,
        'angry': 1,
        'contempt': 2,
        'disgust': 3,
        'fear': 4,
        'happy': 5,
        'sad': 6,
        'surprise': 7
    }
    data = []

    # Iterate through each subfolder
    for emotional_state in emotional_states:

        # Full path of the subfolder
        subfolder_path = os.path.join(dataset_path, emotional_state)

        # Get names of files in the subfolder
        file_names = os.listdir(subfolder_path)

        # Iterate through each file in the subfolder
        for file_name in tqdm(file_names, desc="Processing FER2013", unit="image"):
            full_file_path = os.path.join(subfolder_path, file_name)
            image = cv2.imread(str(full_file_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # write the output image to disk
            new_file_path = os.path.join(destination_path, file_name)
            resized = imutils.resize(gray, width=48, height=48)
            cv2.imwrite(new_file_path, resized)

            # Write the information in the dataframe
            data.append({'emotion': mapping[emotional_state], 'corresponding_image': new_file_path})

    # Writes the data in the csv file
    temp_df = pd.DataFrame(data=data)
    temp_df.to_csv(emotion_csv_file, index=False, header=False, mode='a')


# Loading the parameters
params_file = sys.argv[1]
with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

dataset_images_augmented_path = params['data_augmentation']['dataset_images_augmented_path']
dataset_images_augmented_emotions_path = params['data_augmentation']['dataset_emotions_path']
dataset_images_fer2013_path = params['normalization']['fer_2013_path']
normalization_path = params['normalization']['normalization_path']
emotion_csv_path = params['normalization']['csv_path']


Path(normalization_path).mkdir(parents=True, exist_ok=True)

# Normalize CK+ dataset and creates the csv of emotions
ck_normalization(dataset_images_augmented_path)
ck_emotional_state_normalization(dataset_images_augmented_emotions_path, dataset_images_augmented_path, normalization_path, emotion_csv_path)

# Normalize FER 2013 datasets
fer2013_normalization(dataset_images_fer2013_path + "/train", normalization_path, normalization_path + "/emotions.csv")
fer2013_normalization(dataset_images_fer2013_path + "/test", normalization_path, normalization_path + "/emotions.csv")
