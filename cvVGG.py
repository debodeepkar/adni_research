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

#from keras import backend as K
#K.set_image_dim_ordering('th')

def get_model_name(k):
    return 'model_'+str(k)+'.h5'

VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

save_dir = '/home/ubuntu/debo/saved_models/'
fold_var = 1
train_data = pd.read_csv('train.csv')

#X = train_data[['file']]
Y = train_data[['label']]

kf = KFold(n_splits = 5)

#skf = StratifiedKFold(n_split = 5, random_state = 7, shuffle = True)

train_path = "/home/ubuntu/debo/train"

for train_index, val_index in kf.split(np.zeros(735),Y):
    training_data = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]

    train_data_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input).flow_from_dataframe(training_data, directory = train_path,target_size=(224,224), x_col = "FILE", y_col = "label",class_mode = "categorical", shuffle = True, batch_size=10)
    valid_data_generator  = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input).flow_from_dataframe(validation_data, directory = train_path,target_size=(224,224), x_col = "FILE", y_col = "label",class_mode = "categorical", shuffle = True,batch_size=10)


    # CREATE NEW MODEL
    model = tf.keras.applications.VGG19(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=2)

    # COMPILE NEW MODEL
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

    # CREATE CALLBACKS
    epoch = "epochs/"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), monitor='val_acc')
    #early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15)
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+epoch+"save_at_{epoch}.h5", early_stopping)
    callbacks_list = [checkpoint]
    
    # There can be other callbacks, but just showing one because it involves the model name
    # This saves the best model
    # FIT THE MODEL
    history = self.fit(train_data_generator,epochs=5,  validation_data=valid_data_generator, verbose = 2)

    # LOAD BEST MODEL to evaluate the performance of the model
    self.save_weights("/home/ubuntu/debo/saved_models/model_"+str(fold_var)+".h5")
    self.load_weights("/home/ubuntu/debo/saved_models/model_"+str(fold_var)+".h5")
    results = self.evaluate(valid_data_generator)
    results = dict(zip(self.metrics_names,results))
    VALIDATION_ACCURACY.append(results['acc'])
    VALIDATION_LOSS.append(results['loss'])

    tf.keras.backend.clear_session()
    fold_var += 1


