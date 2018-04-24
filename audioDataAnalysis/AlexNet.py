from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


def get_alexnet2():
    model = Sequential()
    model.add(Convolution2D(96, (11, 11), border_mode='same', input_shape=(227, 227, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(128, (5, 5), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(192, (3, 3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(384, (3, 3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(4096, init='normal', input_shape=(12 * 12 * 256,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4096, init='normal', input_shape=(4096,), name='dense_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1000, init='normal', input_shape=(4096,)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model