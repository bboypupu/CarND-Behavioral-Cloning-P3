#!/usr/bin/env python
# coding: utf-8
#! /opt/carnd_p3/behavioral/bin/python3
from workspace_utils import active_session
import os
import subprocess

with active_session():
    os.system("/opt/carnd_p3/linux_sim/linux_sim.x86_64")
    print("start long-running work")
   
    # In[1]:


    import csv
    import cv2
    import numpy as np

    samples = []
    with open('../new_data/driving_log.csv') as file:
        reader = csv.reader(file)
        next(reader)
        for line in reader:
            samples.append(line)
    print("Done reading csv file")


    # In[2]:


    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)


    # In[3]:


    import sklearn
    from random import shuffle

    correction = 0.2

    def load_image(index, sample):
        path = '../new_data/IMG/'+sample[index].split('/')[-1]
        return cv2.imread(path)

    def flip(image):
        return cv2.flip(image, 1)

    def generator(samples, batch_size=32):
        num_samples = len(samples)
        while True: # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    center_image = load_image(0, batch_sample)
                    center_angle = float(batch_sample[3])
                    left_image = load_image(1, batch_sample)
                    left_angle = center_angle + correction
                    right_image = load_image(2, batch_sample)
                    right_angle = center_angle - correction

                    images.extend([center_image, left_image, right_image,
                                   flip(center_image), flip(left_image), flip(right_image)])
                    angles.extend([center_angle, left_angle, right_angle,
                                   center_angle*-1.0, left_angle*-1.0, right_angle*-1.0])

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    # Set our batch size
    batch_size=32

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    ch, row, col = 3, 80, 320  # Trimmed image format




    # In[4]:


    from keras.models import Sequential, Model
    from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Conv2D, MaxPooling2D
    from keras.layers.convolutional import Convolution2D
    import matplotlib.pyplot as plt


    # In[5]:


    # Build the model
    print("Building the model")
    keep_prob = 0.8
    epochs = 3

    model = Sequential()

    # Normalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

    # Crop ROI
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # Convolutional layers
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    
    model.add(Dropout(keep_prob))
    
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))


    # In[6]:
#     print("Reload model")
#     from keras.models import load_model
#     model = load_model("model.h5")

    print("Start training")
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=epochs, verbose=1)
    model.save('model.h5')


    # In[ ]:


    print(history_object.history.keys())
    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])


    # In[ ]:


#     plt.plot(history_object.history['loss'])
#     plt.plot(history_object.history['val_loss'])
#     plt.title('model mean squared error loss')
#     plt.ylabel('mean squared error loss')
#     plt.xlabel('epoch')
#     plt.legend(['training set', 'validation set'], loc='upper right')
#     plt.show()

