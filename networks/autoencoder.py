from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras import backend as K


def get_autoencoder(input_shape):

    input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format

    # nn.add(Conv2D(64, kernel_size=(7, 7), activation='relu', input_shape=input_size))
    # nn.add(MaxPooling2D(pool_size=(2, 2)))
    # nn.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
    #
    # nn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    x = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    print("shape of encoded", K.int_shape(encoded))

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    print("shape of decoded", K.int_shape(decoded))
    return Model(input_img, decoded)


def sequential_autoencoder(input_shape):

    # input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
    nn = Sequential()

    nn.add(Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same', input_shape = input_shape))
    nn.add(MaxPooling2D((2, 2), padding='same'))
    nn.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    nn.add(MaxPooling2D((2, 2), padding='same'))
    nn.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    nn.add(MaxPooling2D((2, 2), padding='same'))

    nn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    nn.add(UpSampling2D((2, 2)))
    nn.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    nn.add(UpSampling2D((2, 2)))
    nn.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    nn.add(UpSampling2D((2, 2)))
    nn.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    return nn
