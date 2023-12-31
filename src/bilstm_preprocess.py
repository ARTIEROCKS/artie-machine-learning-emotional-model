import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from pathlib import Path
import pandas as pd
from mtcnn import MTCNN
import cv2
import numpy as np

import sys
import warnings
from IPython.display import display
import yaml
from tqdm import tqdm


# 1- Getting images and emotional_state from a path
def get_images(image_path, emotional_state_path, image_extension='png', emotional_state_extension='txt'):
    print("Image List and Emotion Collection")
    image_list = list(Path(image_path).rglob("*" + image_extension))
    emotions_file_list = list(Path(emotional_state_path).rglob("*" + emotional_state_extension))

    emotions_list = []
    corresponding_images = []

    for emotion_file in emotions_file_list:
        # Reading the emotional state from the emotions_file_list
        f = open(str(emotion_file), "r")
        contents = f.read()
        value = float(contents)
        emotion = int(value)
        emotions_list.append(emotion)

        # Gets the corresponding image name
        emotion_file_splitted = str(emotion_file).split("/")
        emotion_file_name = emotion_file_splitted[len(emotion_file_splitted) - 1]
        emotion_file_name_splitted = emotion_file_name.split(".")
        emotion_file_name_splitted = emotion_file_name_splitted[0].split("_emotion")
        emotion_file_name = emotion_file_name_splitted[0]
        corresponding_images.append(emotion_file_name)

    # Adds all the rows and columns to the emotion data frame
    d = {'emotion': emotions_list, 'corresponding_image': corresponding_images}
    emotions_df = pd.DataFrame(data=d)

    return image_list, emotions_df


# 2- Face detection using MTCNN
def face_detection(image, mtcnn):
    path = str(image)
    # print("Detecting faces for: " + path)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    face = mtcnn.detect_faces(img)

    return face, img


# 3- Data augmentation doing some rotations and flips
def data_augmentation(image, face_coordinates, angles=(-7.5, -5, -2.5, 2.5, 5, 7.5)):
    images_augmented = []
    scale = 1.0

    # 1- Gets the face from the image
    # print("Data augmentation for the face")
    bounding_box = face_coordinates[0]['box']
    face = image[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]

    # 2- Flips horizontally
    # print("Flip the image horizontally")
    face_flipped = cv2.flip(face, 1)
    images_augmented.append(face)
    images_augmented.append(face_flipped)

    # cv2_imshow(image_flipped)

    # 3- Rotates the normal and the flipped images
    # print("Rotates the image")
    for angle in angles:
        # get image height, width and calculates the center of the image
        (h, w) = face.shape[:2]
        center = (w / 2, h / 2)

        # Rotates the image
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_face = cv2.warpAffine(face, M, (h, w))

        # Adds the images rotated in the result array
        images_augmented.append(rotated_face)

        del rotated_face

    return images_augmented


# 4- Function that resizes an image and makes the borders black
def image_resize(face, target_size=(256, 256, 3)):
    # 2.1- Transforms the face with a maximum of 256 pixels
    if face.shape[0] >= face.shape[1] and face.shape[0] > target_size[0]:
        # Calculates the proportion
        prop = target_size[0] / face.shape[0]

        # Resizes the image
        res = cv2.resize(face, dsize=(int(face.shape[1] * prop), target_size[0]), interpolation=cv2.INTER_CUBIC)

    else:
        if face.shape[1] >= face.shape[1] > target_size[1]:
            # Calculates the proportion
            prop = target_size[1] / face.shape[1]

            # Resizes the image
            res = cv2.resize(face, dsize=(target_size[1], int(face.shape[0] * prop)), interpolation=cv2.INTER_CUBIC)
        else:
            res = face

    # 2.2- Creates a 299x299 array
    delta_w = target_size[1] - res.shape[1]
    delta_h = target_size[0] - res.shape[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    del res

    return new_im


# 5- Converter fom numpy array
def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))


# Loading the parameters
params_file = sys.argv[1]

with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

# np.set_printoptions(threshold=sys.maxsize)
warnings.simplefilter('ignore')
dataset_images_path = params['data_augmentation']['dataset_images_path']
dataset_emotions_path = params['data_augmentation']['dataset_emotions_path']
dataset_emotions_augmented_path = params['data_augmentation']['dataset_emotions_augmented_path']
dataset_images_augmented_path = params['data_augmentation']['dataset_images_augmented_path']
detector = MTCNN()

Path(dataset_emotions_augmented_path).mkdir(parents=True, exist_ok=True)
Path(dataset_images_augmented_path).mkdir(parents=True, exist_ok=True)

# 1- Getting images and emotional states
image_list, emotions_df = get_images(dataset_images_path, dataset_emotions_path)

# 2- Performing the preprocessing
emotion_list = []
augmented_file_list = []
images = []
element = False

for image_path in tqdm(image_list, desc="Procesando", unit="imagen"):

    # 2.1- Gets the new path and the filename without the extension
    new_path = str(image_path).replace(dataset_images_path, dataset_images_augmented_path)
    new_path_arr = new_path.split("/")
    file_name = new_path_arr[len(new_path_arr) - 1]
    file_name_arr = file_name.split(".")

    file_name = file_name_arr[0]
    # print(file_name)

    # 2.2- Gets the emotional state corresponding to the file
    df = emotions_df.loc[emotions_df['corresponding_image'] == file_name]

    emotion = int(-1)

    # If there is a result: corresponding emotion, if not: -1
    if not df.empty:
        # If there is a result, we get the corresponding emotion
        # display(df['emotion'].values[0])
        emotion = df['emotion'].values[0]

    # 2.4- Data augmentation
    images_augmented = []
    if emotion >= 0:
        face, img = face_detection(image_path, detector)
        images_augmented = data_augmentation(img, face)
        del img, face

    del df

    for i in range(len(images_augmented)):

        # 2.5.1- Creates the new filename and the directory
        new_file_name = file_name + "_" + str(i)
        new_file_full_path = new_path.replace(file_name, new_file_name)
        new_file_path = new_file_full_path.replace(new_file_name + '.png', '')

        # If the file exists we continue to the next file
        if os.path.exists(new_file_full_path):
            continue

        # 2.5.2- Creates the full directory
        Path(new_file_path).mkdir(parents=True, exist_ok=True)

        # 2.5.3- Adds the augmented file name and the corresponding emotion
        emotion_list.append(emotion)
        augmented_file_list.append(new_file_full_path)

        # 2.5.4- Resizes the image
        image_resized = image_resize(images_augmented[i])

        # 2.5.4- Writes the image in the file system and adds the image to the list
        np_image = np.asarray(image_resized)
        images.append(np_image)
        cv2.imwrite(new_file_full_path, image_resized)
        del image_resized, np_image

        # print("FILE: ", new_file_full_path, "EMOTION: ", str(emotion))

    del images_augmented

    # 3- Saves the all the emotions with their corresponding augmented file list
    d = {'emotion': emotion_list, 'corresponding_image': augmented_file_list}
    temp_df = pd.DataFrame(data=d)
    if element == False:
        temp_df.to_csv(dataset_emotions_augmented_path + "/emotions.csv", index=True, header=True, mode='w')
        element = True
    else:
        temp_df.to_csv(dataset_emotions_augmented_path + "/emotions.csv", index=True, header=False, mode='a')

    del emotion_list, augmented_file_list, images, temp_df
    emotion_list = augmented_file_list = images = []

del detector
