import tensorflow as tf
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Dropout
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.regularizers import l2
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['AD', 'NC'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=['AD', 'NC'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['AD', 'NC'], batch_size=10, shuffle=False)

self = Sequential()
self.add(Conv2D(64, kernel_size=(3,3), padding= 'same',activation= 'relu', input_shape= (224,224,3)))
self.add(Conv2D(64, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

self.add(Conv2D(128, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(Conv2D(128, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

self.add(Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

self.add(Flatten())
self.add(Dense(4096, kernel_regularizer=l2(0.0001), activation= 'relu'))
self.add(Dropout(0.1))
self.add(Dense(4096,kernel_regularizer=l2(0.0001), activation= 'relu'))
self.add(Dropout(0.1))
self.add(Dense(2, activation= 'softmax'))
self.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['acc'])


self.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=epoch,
          verbose=2
)

predictions = model1.predict(x=test_batches, steps=len(test_batches), verbose=0)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
print(cm)
vgg=accuracy_score(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
print(vgg)
