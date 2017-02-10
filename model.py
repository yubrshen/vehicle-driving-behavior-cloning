from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
import math

ch, row, col = 3, 160, 320  # camera format
# new image dimension
new_row, new_col = 32, 128

def steering_model(optimizer = optimizers.Adam(lr=1e-04), loss='mse', metrics = ['accuracy']):
    """
    returns a Keras network model
    """
    model = Sequential()
    model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(new_row, new_col, 3), activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(core.Flatten())
    model.add(core.Dense(300, activation='relu'))
    model.add(core.Dropout(.5))
    model.add(core.Dense(100, activation='relu'))
    model.add(core.Dropout(.25))
    model.add(core.Dense(20, activation='relu'))
    model.add(core.Dense(1))
 
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    return model
