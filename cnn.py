import matplotlib
matplotlib.use("Agg")
# Part 1 - Building the CNN
NUM_EPOCHS = 50
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
tf.test.gpu_device_name()

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import pathlib


import numpy as np
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix


classifier = tf.keras.Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 11, activation = 'softmax'))




# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()
# Part 2 - Fitting the CNN to the images


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')


#checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
 #   save_best_only=True, mode='auto', period=1)


H = classifier.fit_generator(training_set,
                             steps_per_epoch = 12192,
                             epochs = NUM_EPOCHS,
                             validation_data = test_set,
                             validation_steps = 2958)
                             #callbacks=[checkpoint])
export_dir = "/tmp/savedModel"
tf.saved_model.saved(classifier, export_dir)
