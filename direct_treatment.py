from audioDataAnalysis.Utils import get_labels, get_files
from Lib import aifc
import numpy as np
from sklearn import svm
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


labels_path = '../KaggleData/train.csv'
save_path = 'results/'
path = '../KaggleData/train/'

labels = get_labels(labels_path)
files = get_files(path)
random.shuffle(files)

y = []
x = []

if __name__ == '__main__':
    for i in files:
        with aifc.open(path + i, 'r') as f:
            nframes = f.getnframes()
            strsig = f.readframes(nframes)
            data = np.fromstring(strsig, np.short).byteswap()
        x.append(data)
        y.append(labels[i])

    x = np.array(x).reshape([len(x), 4000, 1, 1])
    print(x.shape)



    p = 0.2
    train_data = x[:int(len(x) * (1 - p)) // 1,:]
    test_data = x[int(len(x) * (1 - p)) // 1:,:]

    # Normalize data
    x_train = train_data.astype('float32')
    x_test = test_data.astype('float32')
    x_train = (x_train - x.min())/x.max()
    x_test = (x_test - x.min())/x.max()


    train_labels = y[:int(len(y) * (1 - p)) // 1]
    test_target = y[int(len(y) * (1 - p)) // 1:]

    from keras.utils import np_utils
    train_labels = np_utils.to_categorical(train_labels, 2)
    test_target = np_utils.to_categorical(test_target, 2)

    # classifier = svm.SVC(C=0.625)  # Karnowski used C=0.0625
    # classifier.fit(train_data, train_labels)
    # acc = classifier.score(test_data, test_target)
    # print(acc)

    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Conv1D
    from convnetskeras.convnets import convnet
    from keras.optimizers import SGD

    batch_size = 200
    epochs = 12

    nn = Sequential()
    # nn.add(Conv1D(128, 47, activation='relu', input_shape=(256, 4000,)))
    nn.add(Conv2D(128, (11, 1),
                          border_mode="same",
                          activation="relu",
                          input_shape=(4000, 1, 1, )))
    nn.add(Conv2D(64, (23, 1), activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2, 1)))
    nn.add(Flatten())
    nn.add(Dense(2048, activation='relu'))
    nn.add(Dense(2048, activation='relu'))
    nn.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    nn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    history = nn.fit(train_data, train_labels,
        # steps_per_epoch=20,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1)
        # validation_data=(test_data, test_target))

    score = nn.evaluate(test_data, test_target, batch_size=batch_size)

    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])