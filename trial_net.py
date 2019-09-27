from __future__ import division
import keras
import os
from keras.preprocessing.image import ImageDataGenerator
import time

print('Using Keras version', keras.__version__)

save_path = 'results/'
tag = 'low32_now'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_size = [32, 32, 1]
batch_size = 300


datagen = ImageDataGenerator()
data_generator = datagen.flow_from_directory(
    '../KaggleData/DIYS/Train',
    batch_size=batch_size,
    target_size=input_size[:-1],
    shuffle=True,
    color_mode="grayscale",
    class_mode='categorical')

test_datagen = ImageDataGenerator()
test_gen = test_datagen.flow_from_directory(
    '../KaggleData/DIYS/Test',
    batch_size=batch_size,
    target_size=input_size[:-1],
    shuffle=False,
    color_mode="grayscale",
    class_mode='categorical')

from networks.tf_nets import get_nn1, get_nn32_1

nn = get_nn32_1(input_size)

#adagrab worse in test


opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# nn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy']) #default

nn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = nn.fit_generator(data_generator,
                           samples_per_epoch=data_generator.samples,
                           # validation_data=test_gen,
                           nb_val_samples=3000,
                           epochs=150,
                           verbose=1)


score = nn.evaluate_generator(test_gen, steps=20)

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
plt.savefig(save_path + tag + 'model_accuracy' + str(score[1]) + '.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig(save_path + tag + 'model_loss' + str(score[1]) + '.pdf')

# datagen = ImageDataGenerator()
# data_generator = datagen.flow_from_directory(
#     '../KaggleData/DIYS2/Train',
#     batch_size=batch_size,
#     target_size=input_size[:-1],
#     shuffle=False,
#     color_mode="grayscale",
#     class_mode='categorical')
#
#
# # Confusion Matrix
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
# # Compute probabilities
# y_train = data_generator.classes
# Y_pred = nn.predict_generator(data_generator, steps=80)
# # print(Y_pred.shape)
# # Assign most probable label
# y_pred = np.argmax(Y_pred, axis=1)
# # Plot statistics
# print('Analysis of results for Training')
# target_names = ['no_whale', 'whale']
# print(classification_report(y_train, y_pred, target_names=target_names))
# print(confusion_matrix(y_train, y_pred))

# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
# Compute probabilities
y_test = test_gen.classes
Y_pred = nn.predict_generator(test_gen, steps=20)
# print(Y_pred.shape)
# Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
# Plot statistics
print('\nAnalysis of results for Test')
target_names = ['no_whale', 'whale']
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred))

y_pred_proba = nn.predict_generator(test_gen, steps=20)[::, 1]
print('\nROC AUC with probabilities %.4f' % roc_auc_score(y, y_pred_proba))


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
# y_pred_proba = winners.model.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='Winners (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(save_path + tag + 'model_auc' + str(score[1]) + '.pdf')
plt.close()

# Saving model and weights
from keras.models import model_from_json
# nn_json = nn.to_json()
# with open(save_path + 'nn_' + tag +str(score[1])+ '.json', 'w') as json_file:
#     json_file.write(nn_json)
# weights_file = "weights_"+ tag +str(score[1])+".hdf5"
# nn.save_weights('../weights/' + weights_file, overwrite=True)

# Loading model and weights
# json_file = open(save_path + 'nn' + str(score[1]) + '.json','r')
# nn_json = json_file.read()
# json_file.close()
# nn = model_from_json(nn_json)
# nn.load_weights(save_path + weights_file)
