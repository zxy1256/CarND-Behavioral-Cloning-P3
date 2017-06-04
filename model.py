import csv
import cv2
import numpy as np

lines = []
with open('../driving_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

def read_image(source_path):
    '''Read an image
    '''
    filename = source_path.split('/')[-1]
    current_path = '../driving_data/IMG/' + filename
    return cv2.imread(current_path)

images = []
measurements = []
correction = 0.2
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../driving_data/IMG/' + filename
    image = read_image(line[0])
    image_left = read_image(line[1])
    image_right = read_image(line[2])
    images.extend([image, image_left, image_right])
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    measurements.extend([steering_center, steering_left, steering_right])

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D


# Implementing pipeline as reported in
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save('model.h5')