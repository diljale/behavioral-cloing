import csv
import cv2
import numpy as np
import sklearn

lines = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.3)

from sklearn.utils import shuffle
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                current_path = "./data/IMG/" + filename
                image = cv2.imread(current_path)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)
                images.append(cv2.flip(image, 1))
                angles.append(angle * -1.0)
                
                correction = 0.3
                source_path = batch_sample[1]
                filename = source_path.split('/')[-1]
                current_path = "./data/IMG/" + filename
                image = cv2.imread(current_path)
                images.append(image)
                left_angle = angle + correction
                angles.append(left_angle)
                images.append(cv2.flip(image, 1))
                angles.append(left_angle * -1.0)

                source_path = batch_sample[2]
                filename = source_path.split('/')[-1]
                current_path = "./data/IMG/" + filename
                image = cv2.imread(current_path)
                images.append(image)
                right_angle = angle - correction
                angles.append(right_angle)
                images.append(cv2.flip(image, 1))
                angles.append(right_angle * -1.0)

    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)
"""

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = "./data/IMG/" + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    correction = 0.3
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = "./data/IMG/" + filename
    image = cv2.imread(current_path)
    images.append(image)
    left_measurement = measurement + correction
    measurements.append(left_measurement)
    
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = "./data/IMG/" + filename
    image = cv2.imread(current_path)
    images.append(image)
    right_measurement = measurement - correction
    measurements.append(right_measurement)
 
X_train = np.array(images)
y_train = np.array(measurements)
"""

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

"""
I am using NVIDIA's CNN, but modified it by adding dropout since i am using big training set to avoid overfitting
"""
model = Sequential()
model.add(Lambda(lambda x: (x / 127.0) - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(200))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
model.fit_generator(train_generator, samples_per_epoch= 6*len(train_samples), validation_data=validation_generator, nb_val_samples=len(6*validation_samples), nb_epoch=3)

model.save('model.h5')

