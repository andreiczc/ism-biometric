import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def center_pixel_values(x):
    return (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x)


train_datagen = ImageDataGenerator(rescale=1./255., rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255.)

train_dir = 'rsc/Train'
validation_dir = 'rsc/Test'

train_generator = train_datagen.flow_from_directory(
    train_dir, batch_size=32, target_size=(160, 160))

validation_generator = test_datagen.flow_from_directory(
    validation_dir,  batch_size=32, target_size=(160, 160))

inputs = layers.Input(shape=(160, 160, 3))
lambda_1 = layers.Lambda(lambda x: center_pixel_values(x))(inputs)
convo_1 = layers.Conv2D(
    filters=24, kernel_size=5, strides=2, activation='relu')(lambda_1)
convo_2 = layers.Conv2D(
    filters=36, kernel_size=5, strides=2, activation='relu')(convo_1)
convo_3 = layers.Conv2D(
    filters=48, kernel_size=5, strides=2, activation='relu')(convo_2)
convo_4 = layers.Conv2D(
    filters=64, kernel_size=3, strides=1, activation='relu')(convo_3)
convo_5 = layers.Conv2D(
    filters=64, kernel_size=3, strides=1, activation='relu')(convo_4)
flatten = layers.Flatten()(convo_5)
dense_1 = layers.Dense(100)(flatten)
dense_2 = layers.Dense(50)(dense_1)
dense_3 = layers.Dense(10)(dense_2)
predictions = layers.Dense(3, activation='softmax')(dense_3)

model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=Adamax(learning_rate=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(
    train_generator, validation_data=validation_generator, epochs=40)

model.save('./trained_model')
