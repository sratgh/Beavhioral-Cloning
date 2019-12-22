# Imports
import csv
import cv2
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.backend import clear_session

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Samples list for storing strings from the .csv file
samples=[]

# Clear the previous session
clear_session()

# Read in features and labels
with open('../data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        samples.append(row)

# Can be used for quick training check that everything works
#samples = samples[:16]

# Split train and validation set. 20% of all samples will be used for validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Define the generator function
def generator(samples, batch_size=32, training=True):
    num_samples = len(samples)
    while 1:            # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Read in images
                path = '../data/IMG/'
                image_center = cv2.cvtColor(cv2.imread(path + batch_sample[0].split('/')[-1]), cv2.COLOR_BGR2RGB)

                # Append images to list
                if training:
                    image_left = cv2.cvtColor(cv2.imread(path + batch_sample[1].split('/')[-1]), cv2.COLOR_BGR2RGB)
                    image_right = cv2.cvtColor(cv2.imread(path + batch_sample[2].split('/')[-1]), cv2.COLOR_BGR2RGB)
                    images.extend([image_center, image_left, image_right])
                else:
                    images.append(image_center)

                # Read in center measurement and create left and right
                # values with artificial correction offset
                # Finetune correction value
                steering_center = float(batch_sample[3])
                # Add left and right images only in training
                if training:
                    correction = 0.2
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction
                    angles.extend([steering_center, steering_left, steering_right])
                else:
                    angles.append(steering_center)

            # Augment images
            # Augment only for training
            if training:
                augmented_images = []
                augmented_angles = []
                for image, angle in zip(images, angles):
                    augmented_images.append(cv2.flip(image, 1))
                    augmented_angles.append(angle*-1.0)
                images.extend(augmented_images)
                angles.extend(augmented_angles)

            # Turn into numpy arrays and yield shuffled data
            X_data = np.array(images)
            y_data = np.array(angles)
            yield shuffle(X_data, y_data)

# Keras model
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,(5,5),subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,(5,5),subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# INFO: Transfer learning could be used as well

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Compile and train the model using the generator function
# INFO: Batch size from 8 tp 16
train_generator = generator(train_samples, batch_size=8, training=True)
validation_generator = generator(validation_samples, batch_size=8, training=False)

# Train the model and store training history in file
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples),
                                     epochs=4,
                                     verbose=1)

# Save model
model.save('model.h5')

# Save training history
pickle.dump(history_object.history, open( "history.p", "wb" ))

print("Done.")
