import sys
import yaml
import cv2
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.losses import categorical_crossentropy
from keras.src.optimizers import Adam
from tqdm import tqdm


# Relating image and emotional state
def relate_image_emotional_state(emotion_df, normalized_path, n_labels):

    # Setting the class
    indexes = emotion_df['emotion'].values

    # Setting the images values and the classes
    x = []
    y = []

    # Iterate over the values in the "corresponding_image" column
    for index, image_path in tqdm(enumerate(emotion_df['corresponding_image']), desc="Procesando", unit="imagen"):
        parts = image_path.split("/")
        file_name = parts[-1]
        image = cv2.imread(normalized_path + "/" + file_name, cv2.IMREAD_GRAYSCALE)
        x.append(image)

        y_temp = np.zeros(n_labels)
        y_temp[indexes[index]-1] = 1
        y.append(y_temp)

    return x, y


# Function to generate the model
def create_model(n_labels):
    input_size = (48, 48, 1)

    # 1st convolution layer
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(n_labels, activation='softmax'))

    # Compliling the model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model


# Function to split the images and classes for the training
def split_for_training(x_all, y_all, t_percentage):
    # Calculate the split index
    split_index = int(len(x) * t_percentage)

    # Split the array for x
    x_train = x_all[:split_index]
    x_test = x_all[split_index:]

    # Split the array for classes
    y_train = y_all[:split_index]
    y_test = y_all[split_index:]
    return x_train, x_test, y_train, y_test


# Loading the parameters
params_file = sys.argv[1]

with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

# Getting parameters
emotion_csv_file = params['training']['emotion_csv_file']
normalized_images_path = params['training']['normalized_path']

num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width, height = 48, 48
train_percentage = 0.8

# Reading pandas and getting labels and files
df = pd.read_csv(emotion_csv_file)
x, y = relate_image_emotional_state(df, normalized_images_path, num_labels)
X_train, X_test, Y_train, Y_test = split_for_training(x, y, train_percentage)

# Reshaping
X_train = np.array(X_train, 'float32')
X_test = np.array(X_test, 'float32')
Y_train = np.array(Y_train, 'float32')
Y_test = np.array(Y_test, 'float32')

# If the image is grayscale, axis 2 does not exist
axis_tuple_x_train = (0, 1, 2) if X_train.ndim == 3 else (0, 1)
axis_tuple_x_test = (0, 1, 2) if X_test.ndim == 3 else (0, 1)

# Calculate mean and standard deviation
mean_x_train = np.mean(X_train, axis=axis_tuple_x_train)
std_x_train = np.std(X_train, axis=axis_tuple_x_train)

mean_x_test = np.mean(X_test, axis=axis_tuple_x_test)
std_x_test = np.std(X_test, axis=axis_tuple_x_test)

# Normalize
X_train -= mean_x_train
X_train /= std_x_train

X_test -= mean_x_test
X_test /= std_x_test

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# Getting the model
cnn = create_model(7)

# Training the model
cnn.fit(X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test),
        shuffle=True)

# Saving the  model to  use it later on
fer_json = cnn.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
cnn.save_weights("fer.h5")
