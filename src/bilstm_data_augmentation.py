import os

from pathlib import Path
import pandas as pd
from mtcnn import MTCNN
import cv2
import numpy as np
import tensorflow as tf
try:
    import tensorflow_addons as tfa  # optional, GPU rotation if available
except Exception:
    tfa = None
import ast

import sys
import warnings
import yaml
from tqdm import tqdm


# 1- Get images list and emotional states from paths
def get_images(image_path, emotional_state_path, image_extension='png', emotional_state_extension='txt'):
    print("Image List and Emotion Collection")
    image_list = list(Path(image_path).rglob("*" + image_extension))
    emotions_file_list = list(Path(emotional_state_path).rglob("*" + emotional_state_extension))

    emotions_list = []
    corresponding_images = []

    for emotion_file in emotions_file_list:
        # Read emotion value from file
        f = open(str(emotion_file), "r")
        contents = f.read()
        value = float(contents)
        emotion = int(value)
        emotions_list.append(emotion)

        # Derive corresponding image base name
        emotion_file_splitted = str(emotion_file).split("/")
        emotion_file_name = emotion_file_splitted[len(emotion_file_splitted) - 1]
        emotion_file_name_splitted = emotion_file_name.split(".")
        emotion_file_name_splitted = emotion_file_name_splitted[0].split("_emotion")
        emotion_file_name = emotion_file_name_splitted[0]
        corresponding_images.append(emotion_file_name)

    # Assemble dataframe mapping emotion -> image
    d = {'emotion': emotions_list, 'corresponding_image': corresponding_images}
    emotions_df = pd.DataFrame(data=d)

    return image_list, emotions_df


# 2- Face detection using MTCNN
def face_detection(image, mtcnn):
    path = str(image)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    face = mtcnn.detect_faces(img)
    return face, img


# 3- Data augmentation (crop, flip, rotations)
# Implemented with TensorFlow (Metal GPU on macOS). If tensorflow-addons is missing, rotation falls back to OpenCV (CPU).
def _rotate_np(face_np_f32, angle_rad):
    # face_np_f32: float32 [0,1]
    angle_deg = float(angle_rad * (180.0 / np.pi))
    h, w = face_np_f32.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(face_np_f32, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Keep 3 channels if original had them
    if face_np_f32.ndim == 3 and rotated.ndim == 2:
        rotated = np.expand_dims(rotated, axis=-1)
    return rotated.astype(np.float32)


def _rotate_tf(face_f32, angle_rad):
    if tfa is not None:
        return tfa.image.rotate(face_f32, angles=angle_rad, interpolation='bilinear')
    # CPU fallback via OpenCV inside numpy_function
    rotated = tf.numpy_function(_rotate_np, [face_f32, angle_rad], tf.float32)
    rotated.set_shape(face_f32.shape)
    return rotated


def data_augmentation(image, face_coordinates, angles=(-7.5, -5, -2.5, 2.5, 5, 7.5)):
    images_augmented = []
    if face_coordinates is None or len(face_coordinates) == 0:
        return images_augmented

    # Extract face ROI
    bounding_box = face_coordinates[0]['box']
    face_np = image[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]

    # Convert to tensor float32 [0,1]
    face = tf.convert_to_tensor(face_np, dtype=tf.uint8)
    face_f32 = tf.image.convert_image_dtype(face, dtype=tf.float32)

    # Horizontal flip
    face_flipped_f32 = tf.image.flip_left_right(face_f32)

    # Original + flipped (uint8)
    images_augmented.append(tf.image.convert_image_dtype(face_f32, dtype=tf.uint8).numpy())
    images_augmented.append(tf.image.convert_image_dtype(face_flipped_f32, dtype=tf.uint8).numpy())

    # Rotations (only original face)
    for angle_deg in angles:
        angle_rad = angle_deg * (np.pi / 180.0)
        rotated_face_f32 = _rotate_tf(face_f32, angle_rad)
        images_augmented.append(tf.image.convert_image_dtype(rotated_face_f32, dtype=tf.uint8).numpy())

    # Release references
    del face_np, face, face_f32, face_flipped_f32
    return images_augmented


# 4- Resize face to target size with aspect ratio preserved (pad with black)
def image_resize(face, target_size=(256, 256, 3)):
    target_h, target_w = target_size[0], target_size[1]
    face_t = tf.convert_to_tensor(face, dtype=tf.uint8)
    face_f32 = tf.image.convert_image_dtype(face_t, dtype=tf.float32)
    resized_f32 = tf.image.resize_with_pad(face_f32, target_h, target_w, method='bicubic', antialias=True)
    resized_u8 = tf.image.convert_image_dtype(resized_f32, dtype=tf.uint8)
    new_im = resized_u8.numpy()
    del face_t, face_f32, resized_f32, resized_u8
    return new_im


# 5- Convert from stringified numpy array representation
def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))


# Data augmentation for LIRIS (extract faces per frame)
def data_augmentation_liris(videos_path, images_path, mtcnn):
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    videos_list = os.listdir(videos_path)
    for video_file in tqdm(videos_list, desc="Augmenting LIRIS data", unit=" video"):
        if video_file.endswith((".avi", ".mp4")):
            full_video_path = os.path.join(videos_path, video_file)
            cap = cv2.VideoCapture(full_video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                face = mtcnn.detect_faces(frame)
                if face is not None and len(face) > 0:
                    bounding_box = face[0]['box']
                    face_crop = frame[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
                    image_name = f"{os.path.splitext(video_file)[0]}_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.png"
                    image_path = os.path.join(images_path, image_name)
                    cv2.imwrite(image_path, face_crop)
            cap.release()


# Augment CK+ dataset
def data_augmentation_ck(images_list, ds_images_path, ds_images_augmented_path, emotions_dataframe):
    for image_path in tqdm(images_list, desc="Processing", unit="image"):
        new_path = str(image_path).replace(ds_images_path, ds_images_augmented_path)
        new_path_arr = new_path.split("/")
        file_name = new_path_arr[len(new_path_arr) - 1].split(".")[0]
        df = emotions_dataframe.loc[emotions_dataframe['corresponding_image'] == file_name]
        emotion = int(-1)
        if not df.empty:
            emotion = df['emotion'].values[0]
        images_augmented = []
        if emotion >= 0:
            face, img = face_detection(image_path, detector)
            images_augmented = data_augmentation(img, face)
            del img, face
        del df
        for i in range(len(images_augmented)):
            new_file_name = file_name + "_" + str(i)
            new_file_full_path = new_path.replace(file_name, new_file_name)
            new_file_path = new_file_full_path.replace(new_file_name + '.png', '')
            if os.path.exists(new_file_full_path):
                continue
            Path(new_file_path).mkdir(parents=True, exist_ok=True)
            image_resized = image_resize(images_augmented[i])
            np_image = np.asarray(image_resized)
            cv2.imwrite(new_file_full_path, image_resized)
            del image_resized, np_image
        del images_augmented


# Load parameters file
params_file = sys.argv[1]
with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

warnings.simplefilter('ignore')
dataset_images_path = params['data_augmentation']['dataset_ck_images_path']
dataset_emotions_path = params['data_augmentation']['dataset_ck_emotions_path']
dataset_emotions_augmented_path = params['data_augmentation']['dataset_ck_emotions_augmented_path']
dataset_images_augmented_path = params['data_augmentation']['dataset_ck_images_augmented_path']
dataset_liris_videos_path = params['data_augmentation']['dataset_liris_videos_path']
dataset_liris_images_path = params['data_augmentation']['dataset_liris_images_augmented_path']
detector = MTCNN()

Path(dataset_emotions_augmented_path).mkdir(parents=True, exist_ok=True)
Path(dataset_images_augmented_path).mkdir(parents=True, exist_ok=True)

# 1- Collect images and emotions
image_list, emotions_df = get_images(dataset_images_path, dataset_emotions_path)

# 2.1 CK+ augmentation
if params['general']['active_datasets']['ck']:
    data_augmentation_ck(image_list, dataset_images_path, dataset_images_augmented_path, emotions_df)

# 2.2 LIRIS augmentation
if params['general']['active_datasets']['liris']:
    data_augmentation_liris(dataset_liris_videos_path, dataset_liris_images_path, detector)

del detector
