from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image
from tensorflow.keras.applications import EfficientNetB0 as Net
import tensorflow as tf

from datetime import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 48
width = 150
height = 150
epochs = 20
NUM_TRAIN = 2000
NUM_TEST = 1000
dropout_rate = 0.2
input_shape = (height, width, 3)


conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = '/media/amarnath.r/a79edd71-da1d-4f68-82ae-3ba1df252b59/dataset/cat_dog/training/'
validation_dir = '/media/amarnath.r/a79edd71-da1d-4f68-82ae-3ba1df252b59/dataset/cat_dog/testing/'

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to target height and width.
        target_size=(height, width),
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
# model.add(layers.Flatten(name="flatten"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(layers.Dense(2, activation='softmax', name="fc_out"))

model.summary()

print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])


checkpoint_filepath = "/home/amarnath.r/amarnath/cop/visualization/weights-improvement-{epoch:02d}.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_accuracy', mode='min', save_best_only=False)
now = datetime.now()
log_dir = '/home/amarnath.r/amarnath/cop/visualization/logs/' + 'eff_{}_{}_{}_{}'.format(now.day,now.month,now.hour,now.minute)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq = 1)


callbacks=[model_checkpoint_callback, tensorboard_callback]

hist = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, verbose=1, callbacks=callbacks)
