from __future__ import division
from networks.autoencoder import get_autoencoder, sequential_autoencoder
import keras
import os
from keras.preprocessing.image import ImageDataGenerator
import time

print('Using Keras 33version', keras.__version__)

save_path = 'autoencoder_results/'
tag = 'autoenccoder_'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_size = [128, 128, 1]
batch_size = 300


datagen = ImageDataGenerator()
data_generator = datagen.flow_from_directory(
    '../KaggleData/DIYS128/Train/',
    batch_size=batch_size,
    class_mode='input',
    target_size=input_size[:-1],
    shuffle=True,
    color_mode="grayscale")

val_datagen = ImageDataGenerator()
val_gen = datagen.flow_from_directory(
    '../KaggleData/DIYS128/Validation/',
    batch_size=batch_size,
    class_mode='input',
    target_size=input_size[:-1],
    shuffle=False,
    color_mode="grayscale")

# train_generator = zip(data_generator, data_generator)
# validation_generator = zip(val_gen, val_gen)

autoencoder = sequential_autoencoder(input_size)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# nn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy']) #default
# autoencoder.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = autoencoder.fit_generator(data_generator,
                        samples_per_epoch=7000,
                        validation_data=val_gen,
                        validation_steps=10,
                        nb_epoch=9,
                        verbose=2)


score = autoencoder.evaluate_generator(val_gen, steps=10)

print(score)


##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.savefig(save_path + tag + 'model_accuracy' + str(score[1]) + '.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig(save_path + tag + 'model_loss' + str(score[1]) + '.pdf')



datagen = ImageDataGenerator()
data_generator = datagen.flow_from_directory(
    '../KaggleData/DIYS2/Train',
    batch_size=batch_size,
    target_size=input_size[:-1],
    shuffle=False,
    color_mode="grayscale",
    class_mode='categorical')


#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
#Compute probabilities
y_train = data_generator.classes
Y_pred = autoencoder.predict_generator(data_generator, steps=70)
# print(Y_pred.shape)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print('Analysis of results')
target_names = ['no_whale', 'whale']
print(classification_report(y_train, y_pred, target_names=target_names))
print(confusion_matrix(y_train, y_pred))

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
#Compute probabilities
y_test = val_gen.classes
Y_pred = autoencoder.predict_generator(val_gen, steps=10)
# print(Y_pred.shape)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print('Analysis of results')
target_names = ['no_whale', 'whale']
print(classification_report(y_test, y_pred,target_names=target_names))
print(confusion_matrix(y_test, y_pred))


#Saving model and weights
from keras.models import model_from_json
nn_json = autoencoder.to_json()
with open(save_path + 'nn_' + tag +str(score[1])+ '.json', 'w') as json_file:
    json_file.write(nn_json)
weights_file = "weights_"+ tag +str(score[1])+".hdf5"
autoencoder.save_weights(save_path + weights_file, overwrite=True)

#Loading model and weights
# json_file = open(save_path + 'nn' + str(score[1]) + '.json','r')
# nn_json = json_file.read()
# json_file.close()
# nn = model_from_json(nn_json)
# nn.load_weights(save_path + weights_file)