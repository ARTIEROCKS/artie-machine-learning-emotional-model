import yaml
import sys
import cv2
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf

# -----------------------------
# CK+ emotion CSV normalization
# -----------------------------

def ck_emotional_state_normalization(emotional_state_path, augmented_images_path, root_path, csv_path):
    """Build a CSV mapping each augmented image to its emotion label for CK+ dataset."""
    print("Building CK+ emotion mapping")

    emotions_list = []
    corresponding_images = []
    emotions_file_list = list(Path(emotional_state_path).rglob("*.txt"))

    for emotion_file in tqdm(emotions_file_list, desc="Normalizing CK+ emotions", unit=" file"):
        # Read emotion value
        with open(str(emotion_file), "r") as f:
            contents = f.read()
        value = float(contents)
        emotion = int(value)

        # Derive base image name from emotion filename
        emotion_file_splitted = str(emotion_file).split("/")
        emotion_file_name = emotion_file_splitted[-1]
        emotion_file_name_splitted = emotion_file_name.split(".")[0].split("_emotion")
        emotion_file_name = emotion_file_name_splitted[0]

        # Find all augmented images that start with the base name
        emotion_image_list = list(Path(root_path).rglob(emotion_file_name + "*.png"))
        for image in emotion_image_list:
            emotions_list.append(emotion)
            corresponding_images.append(str(image))

    # Create DataFrame and append to CSV (header written once by using header=True here)
    d = {'emotion': emotions_list, 'corresponding_image': corresponding_images}
    emotions_df = pd.DataFrame(data=d)
    emotions_df.to_csv(csv_path, index=False, header=True, mode='a')


# ---------------------------------
# CK+ image normalization to 48x48
# ---------------------------------

def ck_normalization(ds_images_augmented_path):
    """Normalize CK+ augmented images: grayscale + resize to 48x48 (GPU resize)."""
    image_list = list(Path(ds_images_augmented_path).rglob("*.png"))
    print("CK+ image normalization starting")

    for input_path in tqdm(image_list, desc="Normalizing CK+", unit=" image"):
        # Load BGR image
        image = cv2.imread(str(input_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # TensorFlow GPU resize
        gray_tensor = tf.convert_to_tensor(gray, dtype=tf.uint8)
        gray_tensor = tf.expand_dims(gray_tensor, axis=-1)  # (H,W,1)
        gray_f32 = tf.image.convert_image_dtype(gray_tensor, dtype=tf.float32)
        resized_f32 = tf.image.resize(gray_f32, (48, 48), method='bilinear', antialias=True)
        resized_u8 = tf.image.convert_image_dtype(resized_f32, dtype=tf.uint8)
        resized = resized_u8.numpy().squeeze(axis=-1)

        cv2.imwrite(os.path.join(normalization_path, str(input_path).split("/")[-1]), resized)


# ---------------------------------
# FER2013 dataset normalization
# ---------------------------------

def fer2013_normalization(dataset_path, destination_path, emotion_csv_file, es_mapping):
    """Normalize FER2013 images (grayscale + 48x48 GPU resize) and append emotion CSV rows."""
    print("FER2013 image normalization starting")

    emotional_states = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    data = []

    for emotional_state in emotional_states:
        subfolder_path = os.path.join(dataset_path, emotional_state)
        file_names = os.listdir(subfolder_path)
        for file_name in tqdm(file_names, desc=f"Normalizing FER2013:{emotional_state}", unit=" image"):
            full_file_path = os.path.join(subfolder_path, file_name)
            image = cv2.imread(str(full_file_path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # GPU resize
            gray_tensor = tf.convert_to_tensor(gray, dtype=tf.uint8)
            gray_tensor = tf.expand_dims(gray_tensor, -1)
            gray_f32 = tf.image.convert_image_dtype(gray_tensor, dtype=tf.float32)
            resized_f32 = tf.image.resize(gray_f32, (48, 48), method='bilinear', antialias=True)
            resized_u8 = tf.image.convert_image_dtype(resized_f32, dtype=tf.uint8)
            resized = resized_u8.numpy().squeeze(-1)

            new_file_path = os.path.join(destination_path, file_name)
            cv2.imwrite(new_file_path, resized)
            data.append({'emotion': int(es_mapping[emotional_state]), 'corresponding_image': new_file_path})

    # Append CSV rows (no header)
    temp_df = pd.DataFrame(data=data)
    temp_df.to_csv(emotion_csv_file, index=False, header=False, mode='a')


# ---------------------------------
# LIRIS dataset normalization
# ---------------------------------

def liris_normalization(source_path, destination_path, emotion_csv_file, es_mapping):
    """Normalize LIRIS images (grayscale + 48x48 GPU resize) and append emotion CSV rows."""
    print("LIRIS image normalization starting")

    image_list = [name for name in os.listdir(source_path) if name.lower().endswith((".png", ".jpg", ".jpeg"))]
    data = []

    for image_name in tqdm(image_list, desc="Normalizing LIRIS", unit=" image"):
        # Extract emotion token from filename (expected pattern with underscore)
        parts = image_name.split('_')
        if len(parts) < 2:
            continue
        emotional_state = parts[1]

        full_file_path = os.path.join(source_path, image_name)
        image = cv2.imread(str(full_file_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # GPU resize
        gray_tensor = tf.convert_to_tensor(gray, dtype=tf.uint8)
        gray_tensor = tf.expand_dims(gray_tensor, -1)
        gray_f32 = tf.image.convert_image_dtype(gray_tensor, dtype=tf.float32)
        resized_f32 = tf.image.resize(gray_f32, (48, 48), method='bilinear', antialias=True)
        resized_u8 = tf.image.convert_image_dtype(resized_f32, dtype=tf.uint8)
        resized = resized_u8.numpy().squeeze(-1)

        new_file_path = os.path.join(destination_path, image_name)
        cv2.imwrite(new_file_path, resized)

        key = emotional_state.lower()
        if key in es_mapping:
            data.append({'emotion': int(es_mapping[key]), 'corresponding_image': new_file_path})

    temp_df = pd.DataFrame(data=data)
    temp_df.to_csv(emotion_csv_file, index=False, header=False, mode='a')


# -----------------------------
# Parameter loading and execution
# -----------------------------

params_file = sys.argv[1]
with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

dataset_ck_images_augmented_path = params['data_augmentation']['dataset_ck_images_augmented_path']
dataset_ck_emotions_augmented_path = params['data_augmentation']['dataset_ck_emotions_augmented_path']
dataset_images_augmented_liris_path = params['data_augmentation']['dataset_liris_images_augmented_path']

dataset_images_fer2013_path = params['normalization']['fer_2013_path']
normalization_path = params['normalization']['normalization_path']
emotion_csv_path = params['normalization']['csv_path']

Path(normalization_path).mkdir(parents=True, exist_ok=True)

# Emotion mapping dictionary (unified across datasets)
mapping = {
    'neutral': 0,
    'angry': 1,
    'anger': 1,
    'contempt': 2,
    'confusing': 2,
    'disgust': 3,
    'fear': 4,
    'happy': 5,
    'sad': 6,
    'surprise': 7,
    'suprise': 7
}

# CK+ normalization + emotions CSV
if params['general']['active_datasets']['ck']:
    ck_normalization(dataset_ck_images_augmented_path)
    ck_emotional_state_normalization(dataset_ck_emotions_augmented_path, dataset_ck_images_augmented_path,
                                     normalization_path, emotion_csv_path)

# FER2013 normalization (train + test)
if params['general']['active_datasets']['fer2013']:
    fer2013_normalization(os.path.join(dataset_images_fer2013_path, "train"), normalization_path, os.path.join(normalization_path, "emotions.csv"), mapping)
    fer2013_normalization(os.path.join(dataset_images_fer2013_path, "test"), normalization_path, os.path.join(normalization_path, "emotions.csv"), mapping)

# LIRIS normalization
if params['general']['active_datasets']['liris']:
    liris_normalization(dataset_images_augmented_liris_path, normalization_path, os.path.join(normalization_path, "emotions.csv"), mapping)
