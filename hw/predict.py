import tensorflow as tf
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def center_pixel_values(x):
    return (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x)


model = tf.keras.models.load_model('./trained_model')
race = ['Negroid', 'Mongoloid', 'Caucasian']

while(True):
    image = cv2.imread('rsc/Train/Mongoloid/111.JPG')
    image = cv2.resize(image, (160, 160))

    imageArr = np.reshape(
        image, (1, image.shape[0], image.shape[1], image.shape[2]))
    prediction = model.predict(imageArr)
    print('logits: {}'.format(prediction))
    prediction = np.argmax(prediction)

    cv2.imshow(race[prediction], image)
    key = cv2.waitKey(5000) & 0xFF

    if key == ord('q'):
        print('stopping')
        break


cv2.destroyAllWindows()
