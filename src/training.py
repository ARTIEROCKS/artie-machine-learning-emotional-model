import sys

import tensorflow as tf
import yaml
import cv2
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.src.losses import categorical_crossentropy
from keras.src.optimizers import Adam
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from enum import Enum

# Enum for creating the model
class Layer(Enum):
    CONV2D = 1
    BATCHNORMALIZATION = 2
    MAXPOOLING2D = 3
    DROPOUT = 4
    FLATTEN = 5
    DENSE = 6


# Relating image and emotional state
def relate_image_emotional_state(emotion_df, normalized_path, n_labels):
    # Setting the class
    indexes = emotion_df['emotion'].values

    # Setting the images values and the classes
    x = []
    y = []

    # Iterate over the values in the "corresponding_image" column
    for index, image_path in tqdm(enumerate(emotion_df['corresponding_image']), desc="Processing", unit=" image"):
        parts = image_path.split("/")
        file_name = parts[-1]
        image = cv2.imread(normalized_path + "/" + file_name, cv2.IMREAD_GRAYSCALE)
        x.append(image)

        y_temp = np.zeros(n_labels)
        y_temp[indexes[index] - 1] = 1
        y.append(y_temp)

    return x, y


# Function to generate the model
def create_model(n_labels, layers=None, values=None):
    if layers is None:
        layers = [Layer.CONV2D, Layer.CONV2D, Layer.MAXPOOLING2D, Layer.DROPOUT,
                  Layer.CONV2D, Layer.CONV2D, Layer.MAXPOOLING2D, Layer.DROPOUT,
                  Layer.CONV2D, Layer.CONV2D, Layer.MAXPOOLING2D, Layer.DROPOUT,
                  Layer.DENSE, Layer.DROPOUT, Layer.DENSE, Layer.DROPOUT, Layer.DENSE]
    if values is None:
        values = [[64,3,3,'relu'],[64,3,3,'relu'],[2,2,2,2],[0.5],
                  [64,3,3,'relu'],[64,3,3,'relu'],[2,2,2,2],[0.5],
                  [128,3,3,'relu'],[128,3,3,'relu'],[2,2,2,2],[0.5],
                  [], [1024, 'relu'], [0.2], [1024, 'relu'], [0.2], [7,'softmax']]


    input_size = (48, 48, 1)

    model = Sequential()
    value_number = 0

    #Performing the convolution layers
    for layer in layers:
        if layer == Layer.CONV2D:
            if value_number > 0:
                model.add(Conv2D(values[value_number][0], kernel_size=(values[value_number][1], values[value_number][2]), activation=values[value_number][3], padding=values[value_number][4]))
            else:
                model.add(Conv2D(values[value_number][0], kernel_size=(values[value_number][1], values[value_number][2]), activation=values[value_number][3],
                                 input_shape=input_size, padding=values[value_number][4]))
        elif layer == Layer.MAXPOOLING2D:
            model.add(MaxPooling2D(pool_size=(values[value_number][0], values[value_number][1]), strides=(values[value_number][2], values[value_number][3])))
        elif layer == Layer.BATCHNORMALIZATION:
            model.add(BatchNormalization())
        elif layer == Layer.DROPOUT:
            model.add(Dropout(values[value_number][0]))
        elif layer == Layer.FLATTEN:
            model.add(Flatten())
        elif layer == Layer.DENSE:
            model.add(Dense(values[value_number][0], activation=values[value_number][1]))

        value_number+=1

    # Compliling the model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(thresholds=0.5),
                           tf.keras.metrics.FalsePositives(),
                           tf.keras.metrics.FalseNegatives(),
                           tf.keras.metrics.TruePositives(),
                           tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.PrecisionAtRecall(0.5),
                           tf.keras.metrics.SensitivityAtSpecificity(0.5),
                           tf.keras.metrics.SpecificityAtSensitivity(0.5),
                           tf.keras.metrics.MeanIoU(
                               n_labels,
                               name=None,
                               dtype=None,
                               ignore_class=None,
                               sparse_y_true=True,
                               sparse_y_pred=True,
                               axis=-1,
                           )
                           ])

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


# Function to plot
def plotting(hist):

    # Saving the plots and metrics
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(hist.history)

    # Convert DataFrame to a list of dictionaries
    metrics_data = {'loss': hist_df['loss'].mean(), 'accuracy': hist_df['accuracy'].mean(),
                    'precision': hist_df['precision'].mean(), 'recall': hist_df['recall'].mean(),
                    'val_loss': hist_df['val_loss'].mean(), 'false_positives': hist_df['false_positives'],
                    'false_negatives': hist_df['false_negatives'].mean(),
                    'true_positives': hist_df['true_positives'].mean(),
                    'true_negatives': hist_df['true_negatives'].mean(),
                    'precision_at_recall': hist_df['precision_at_recall'].mean(),
                    'sensitivity_at_specificity': hist_df['sensitivity_at_specificity'].mean(),
                    'specificity_at_sensitivity': hist_df['specificity_at_sensitivity'].mean(),
                    'mean_io_u': hist_df['mean_io_u']}
    metrics_df = pd.DataFrame.from_records([metrics_data])

    with open(metrics_path + "/plots.csv", mode='w') as f:
        hist_df.to_csv(f, index_label='epoch')

    with open(metrics_path + "/scores.json", mode='w') as f:
        metrics_df.to_json(f)

    # Saving training history plot
    plt.figure(figsize=(19.2, 10.8))
    plt.ylabel('Loss / Accuracy')
    plt.xlabel('Epoch')

    for k in hist_df.keys():
        if k != 'false_positives' and k != 'false_negatives' and k != 'true_positives' and k != 'true_negatives' and k != 'val_false_positives' and k != 'val_false_negatives' and k != 'val_true_positives' and k != 'val_true_negatives':
            plt.plot(history.history[k], label=k)

    plt.legend(loc='best')
    plt.savefig("plots/train_history.png", dpi=100, bbox_inches='tight', pad_inches=0)


# Loading the parameters
params_file = sys.argv[1]

with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

# Getting parameters
emotion_csv_file = params['training']['emotion_csv_file']
normalized_images_path = params['normalization']['normalization_path']
model_path = params['training']['model_path']

num_labels = params['training']['num_labels']
batch_size = params['training']['batch_size']
epochs = params['training']['epochs']
width, height = 48, 48
train_percentage = params['training']['train_percentage']

show_summary = params['training']['show_summary']
metrics_path = params['training']['metrics_path']

# Getting the FCNN structure parameters
structure_layers = [Layer[layer] for layer in params['training']['structure_layers']]
structure_values = params['training']['structure_values']

# Creates the needed directories
Path(model_path).mkdir(parents=True, exist_ok=True)
Path(metrics_path).mkdir(parents=True, exist_ok=True)
Path('plots').mkdir(parents=True, exist_ok=True)

# Reading pandas and getting labels and files
df = pd.read_csv(emotion_csv_file)
x, y = relate_image_emotional_state(df, normalized_images_path, num_labels)
X_train, X_test, Y_train, Y_test = split_for_training(x, y, train_percentage)

# Get the unique shapes of arrays in the list
unique_shapes = set(arr.shape for arr in X_train)

# Print the different sizes
print("Different sizes in the list:")
for shape in unique_shapes:
    print(shape)

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
cnn = create_model(n_labels=num_labels, layers=structure_layers, values=structure_values)

# Training the model
history = cnn.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_test, Y_test),
                  shuffle=True)

# Saving the  model to  use it later on
fer_json = cnn.to_json()
with open(model_path + "/fer.json", "w") as json_file:
    json_file.write(fer_json)
cnn.save(model_path + "/fer.h5")

# If we want to show the summary
if show_summary:
    cnn.summary()
    Path('images').mkdir(parents=True, exist_ok=True)
    tf.keras.utils.plot_model(cnn, to_file='images/model.png', dpi=200)

# Plotting
plotting(history)
