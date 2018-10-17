from audioDataAnalysis.Utils import get_labels, get_files, get_final_image, rgb2gray
from scipy import misc
import keras
import os
import numpy as np

print('Using Keras version', keras.__version__)
labels_path = '../KaggleData/train.csv'
save_path = 'results/'
path = '../KaggleData/Spectrograms/spec_hamming/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

labels = get_labels(labels_path)
files = get_files(path)

input_size = [32, 32, 1]
batch_size = 30

data = np.array([])

for i in files:
    im = misc.imread(path + i)
    final = get_final_image(im, size=[32,32,3])
    final_gray = rgb2gray(final)
    b = final_gray.reshape(32*32,1)
    data = np.concatenate(data, b)

print(data[0:10])



# #Adapt the data as an input of a fully-connected (flatten to 1D)
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
#
# #Normalize data
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train = x_train / 255
# x_test = x_test / 255
#
# #Adapt the labels to the one-hot vector syntax required by the softmax
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)

#Define the NN architecture

from networks.tf_nets import get_nn32_2

nn = get_nn32_2(input_size)

#Model visualization NOT WORKING!
#We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
#from keras.utils import plot_model
#plot_model(nn, to_file=save_path + 'nn.png', show_shapes=True)

nn.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = nn.fit_generator(data_generator,
                        samples_per_epoch=3000,
                        # validation_data=test_gen,
                        nb_val_samples=800,
                        epochs=16,
                        verbose=1)


score = nn.evaluate_generator(test_gen, steps=200)

print(score)

##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig(save_path + '32_model_accuracy' + str(score[1]) + '.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig(save_path + '32_model_loss' + str(score[1]) + '.pdf')

#Saving model and weights
from keras.models import model_from_json
nn_json = nn.to_json()
with open(save_path + 'nngray' +str(score[1])+ '.json', 'w') as json_file:
    json_file.write(nn_json)
weights_file = "weights-gray_32_"+str(score[1])+".hdf5"
nn.save_weights('../weights/' + weights_file, overwrite=True)