from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
# from convnetskeras.convnets import convnet
from keras.optimizers import SGD
import time
#Two hidden layers


def no_conv_nn(input_size):
    nn = Sequential()
    nn.add(Flatten(input_shape=input_size))
    nn.add(BatchNormalization())
    nn.add(Dense(500, activation='relu'))
    nn.add(Dropout(0.5))
    nn.add(BatchNormalization())
    nn.add(Dense(500, activation='relu'))
    nn.add(Dropout(0.5))
    nn.add(Dense(2, activation='softmax'))
    return nn


def get_nn1(input_size):
    nn = Sequential()
    nn.add(Conv2D(64, kernel_size=(7, 7), activation='relu', input_shape=input_size))
    nn.add(BatchNormalization())
    nn.add(MaxPooling2D(pool_size=(4, 2)))
    nn.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
    nn.add(BatchNormalization())
    nn.add(MaxPooling2D(pool_size=(4, 2)))
    nn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    nn.add(BatchNormalization())
    nn.add(MaxPooling2D(pool_size=(4, 2)))
    nn.add(Flatten())
    nn.add(Dense(500, activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dense(500, activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dense(2, activation='softmax'))
    return nn


def winners_nn(input_size):
    nn = Sequential()
    nn.add(Conv2D(20, kernel_size=(7, 7), activation='relu', input_shape=input_size))
    nn.add(Dropout(0.2))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Conv2D(40, kernel_size=(7, 7), activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Flatten())
    nn.add(Dense(512, activation='relu'))
    nn.add(Dropout(0.6))
    nn.add(Dense(2, activation='softmax'))
    return nn


def get_nn32(input_size):
    nn = Sequential()
    nn.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_size))
    nn.add(BatchNormalization())
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Flatten())
    nn.add(Dense(500, activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dense(500, activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dense(2, activation='softmax'))
    return nn


def get_nn32_1(input_size):
    nn = Sequential()
    nn.add(Conv2D(64, kernel_size=(7, 7), activation='relu', input_shape=input_size))
    nn.add(BatchNormalization())
    nn.add(Dropout(0.2))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dropout(0.2))
    nn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dropout(0.2))

    nn.add(Flatten())
    nn.add(Dense(500, activation='relu'))
    nn.add(Dropout(0.5))
    nn.add(BatchNormalization())
    nn.add(Dense(500, activation='relu'))
    nn.add(Dropout(0.5))
    nn.add(BatchNormalization())
    nn.add(Dense(2, activation='softmax'))
    return nn


def get_nn32_3(input_size):
    nn = Sequential()
    nn.add(Conv2D(8, kernel_size=(11, 11), activation='relu', input_shape=input_size))
    nn.add(BatchNormalization())
    # nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Conv2D(16, kernel_size=(9, 9), activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Conv2D(8, kernel_size=(7, 7), activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Flatten())
    nn.add(Dense(500, activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dense(500, activation='relu'))
    nn.add(BatchNormalization())
    nn.add(Dense(2, activation='softmax'))
    return nn


def get_nn32_2(input_size):
    nn = Sequential()
    nn.add(Conv2D(128, kernel_size=(7, 7), activation='relu', input_shape=input_size))
    nn.add(Dropout(0.1))
    nn.add(BatchNormalization())
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    nn.add(Dropout(0.1))
    nn.add(BatchNormalization())
    nn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    nn.add(Dropout(0.1))
    nn.add(BatchNormalization())
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Flatten())
    nn.add(Dense(500, activation='relu'))
    nn.add(Dropout(0.1))
    nn.add(BatchNormalization())
    nn.add(Dense(500, activation='relu'))
    nn.add(Dropout(0.1))
    nn.add(BatchNormalization())
    nn.add(Dense(2, activation='softmax'))
    return nn


# def get_finetune(numb_classes, net='alexnet', freeze=0):
#     sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#     if net == 'alexnet':
#         wp = "../weights/alexnet_weights.h5"
#     elif net == 'vgg_16':
#         wp = "../weights/vgg16_weights.h5"
#     elif net == 'vgg_19':
#         wp = "../weights/vgg19_weights.h5"
#     else:
#         print('Wrong network!')
#         return 0
#     model = convnet(net, weights_path=wp, heatmap=False)
#     model.layers.pop()
#     model.outputs = [model.layers[-1].output]
#     model.layers[-2].outbound_nodes = []
#     x = model.layers[-1].output
#     last = Dense(numb_classes, name='output_layer')(x)
#     prediction = Activation("softmax", name="softmax")(last)
#
#     model = Model(input=model.input, output=prediction)

    # # model.layers.pop()
    # # model.outputs = [model.layers[-1].output]
    # # model.layers[-1].outbound_nodes = []
    # # model.add(Dense(num_class, activation='softmax'))
    #
    # for layer in model.layers[:freeze]:
    #     layer.trainable = False
    # return model, sgd


def get_nn128_3(input_size):
    nn = Sequential()
    nn.add(Conv2D(16, kernel_size=(7, 7), strides=(1, 1), activation='relu', input_shape=input_size))
    nn.add(MaxPooling2D(pool_size=(2, 4)))
    # nn.add(BatchNormalization())

    # nn.add(Conv2D(48, kernel_size=(7, 7), padding="valid", activation='relu'))
    # nn.add(MaxPooling2D(pool_size=(2, 2)))
    # nn.add(BatchNormalization())

    nn.add(Conv2D(32, kernel_size=(5, 5), padding="valid", activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    # nn.add(BatchNormalization())

    nn.add(Flatten())
    nn.add(Dense(512, activation='relu'))
    nn.add(Dropout(0.4))
    # nn.add(BatchNormalization())

    # nn.add(Dense(512, activation='relu'))
    # nn.add(Dropout(0.4))
    # nn.add(BatchNormalization())
    nn.add(Dense(2, activation='softmax'))
    return nn
