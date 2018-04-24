from __future__ import division
from networks.tf_nets import get_finetune
import keras
import os
from keras.preprocessing.image import ImageDataGenerator

print('Using Keras version', keras.__version__)

save_path = 'results/'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_size = [227, 227, 3]
batch_size = 30


datagen = ImageDataGenerator()
data_generator = datagen.flow_from_directory(
    '../KaggleData/DIYS/Train',
    batch_size=batch_size,
    target_size=input_size[:-1],
    # color_mode="grayscale",
    class_mode='categorical')

test_datagen = ImageDataGenerator()
test_gen = test_datagen.flow_from_directory(
    '../KaggleData/DIYS/Test',
    batch_size=batch_size,
    target_size=input_size[:-1],
    shuffle=True,
    # color_mode="grayscale",
    class_mode='categorical')

if __name__ == '__main__':
    model, sgd = get_finetune(2, freeze=0)#27

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    history = model.fit_generator(data_generator,
                                  samples_per_epoch=1500,
                                  # validation_data=validation_generator,
                                  nb_val_samples=800,
                                  nb_epoch=16,
                                  verbose=1)

    score = model.evaluate_generator(test_gen, steps=200)

    ##Store Plots
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Accuracy plot
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(save_path + 'model_accuracy' + str(score[1]) + '.pdf')
    plt.close()
    # Loss plot
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(save_path + 'model_loss' + str(score[1]) + '.pdf')

    # Confusion Matrix
    # from sklearn.metrics import classification_report,confusion_matrix
    # import numpy as np
    # #Compute probabilities
    # Y_pred = nn.predict(x_test)
    # #Assign most probable label
    # y_pred = np.argmax(Y_pred, axis=1)
    # #Plot statistics
    # print('Analysis of results')
    # target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
    # print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

    # Saving model and weights
    from keras.models import model_from_json

    nn_json = model.to_json()
    with open(save_path + 'nngray' + str(score[1]) + '.json', 'w') as json_file:
        json_file.write(nn_json)
    weights_file = "weights-gray_" + str(score[1]) + ".hdf5"
    model.save_weights('../weights/' + weights_file, overwrite=True)

    # Loading model and weights
    # json_file = open(save_path + 'nn' + str(score[1]) + '.json','r')
    # nn_json = json_file.read()
    # json_file.close()
    # nn = model_from_json(nn_json)
