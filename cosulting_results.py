import os
from keras.preprocessing.image import ImageDataGenerator

save_path = 'results/'
tag = 'low32'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_size = [32, 32, 1]
batch_size = 3000


datagen = ImageDataGenerator()
data_generator = datagen.flow_from_directory(
    '../KaggleData/DIYS2/Train',
    batch_size=batch_size,
    target_size=input_size[:-1],
    color_mode="grayscale",
    class_mode='categorical')

val_datagen = ImageDataGenerator()
val_gen = datagen.flow_from_directory(
    '../KaggleData/DIYS2/Validation',
    batch_size=batch_size,
    target_size=input_size[:-1],
    shuffle=False,
    color_mode="grayscale",
    class_mode='categorical')

test_datagen = ImageDataGenerator()
test_gen = test_datagen.flow_from_directory(
    '../KaggleData/DIYS2/Test',
    batch_size=batch_size,
    target_size=input_size[:-1],
    shuffle=False,
    color_mode="grayscale",
    class_mode='categorical')


from keras.models import model_from_json

print('Loading the model.')
json_file = open(save_path + 'nn_low32Dropout0.998333334923.json','r')
nn_json = json_file.read()
json_file.close()
nn = model_from_json(nn_json)

print(nn.summary())

# nn.load_weights('../weights/weights_low32Dropout0.998333334923.hdf5')
# print('Model loaded. Getting the confusion matrices.')
#
# nn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# score = nn.evaluate_generator(test_gen, steps=1)
#
# print(score)
#
#
# #Confusion Matrix
# from sklearn.metrics import classification_report,confusion_matrix
# import numpy as np
# import keras
#
# #Compute probabilities
# y_train = data_generator.classes
# Y_pred = nn.predict_generator(data_generator, steps=7)
# # print(Y_pred.shape)
# #Assign most probable label
# # print(Y_pred)
# # print(np.argmax(Y_pred, axis=1))
# # print(y_train)
# y_pred = np.argmax(Y_pred, axis=1)
# #Plot statistics
# print('Analysis of results')
# target_names = ['no_whale', 'whale']
# print(classification_report(y_train, y_pred, target_names=target_names))
# print(confusion_matrix(y_train, y_pred))
#
#
#
# #Confusion Matrix
# from sklearn.metrics import classification_report,confusion_matrix
# import numpy as np
# from keras import backend as K
# #Compute probabilities
# y_test = test_gen.classes[:3000]
# Y_pred = nn.predict_generator(test_gen, steps=1)
# # print(Y_pred.shape)
# #Assign most probable label
# y_pred = np.argmax(Y_pred, axis=1)
# #Plot statistics
# print('Analysis of results')
# target_names = ['no_whale', 'whale']
# print(classification_report(y_test, y_pred,target_names=target_names))
# print(confusion_matrix(y_test, y_pred))
