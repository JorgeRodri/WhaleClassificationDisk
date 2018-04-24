import pickle
from sklearn import svm
import numpy
from audioDataAnalysis.Utils import get_images_ready_alexnet_batch, split_list
import time
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from keras.models import Model

def extract_features(files, path, labels, num_batches=1, net='alexnet'):
    batches = split_list(files, wanted_parts=num_batches)
    size = len(batches)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    if net == 'alexnet':
        wp = "../weights/alexnet_weights.h5"
        sc = [227, 227, 3]
    elif net == 'vgg_16':
        wp = "../weights/vgg16_weights.h5"
        sc = [224, 224, 3]
    elif net == 'vgg_19':
        wp = "../weights/vgg19_weights.h5"
        sc = [224, 224, 3]
    else:
        print('Wrong network!')
        return 0
    model = convnet(net, weights_path=wp, heatmap=False)
    model.compile(optimizer=sgd, loss='mse')
    feature_map = Model(input=model.input, output=model.get_layer('dense_2').output)
    y=[]

    n = 0
    print('Starting, done: ' + str(n) + '%')
    times = [time.time()]
    for names in batches:
        n += 1
        for case in names:
            y.append(labels[case[:-4]+'.aiff'])
        actual_batch = get_images_ready_alexnet_batch(names, path, size=sc)
        features = feature_map.predict(actual_batch)
        if n == 1:
            data = features
        else:
            data = numpy.concatenate((data, features))
        if n * 100 / size % 5 == 0:
            times.append(time.time())
            print('Done for: ' + str(n * 100 / size) + '%')
            print('Time for the last 5%: ' + str(times[-1] - times[-2]))
            print('Expected time to finish: ' + str((20-n*20/size) * (times[-1] - times[-2]) / 60) + ' minutes.')
    print('Done.')
    return data, y


def __get_all_SVM__():
    with open('features.txt', 'rb') as f:
        data = pickle.load(f)
    with open('labels.txt', 'rb') as f:
        y = pickle.load(f)

    p = 0.2
    train_data = data[:int(len(data) * (1 - p)) // 1,:]
    test_data = data[int(len(data) * (1 - p)) // 1:,:]

    train_labels = y[:int(len(y) * (1 - p)) // 1]
    test_target = y[int(len(y) * (1 - p)) // 1:]

    n = 0
    accs = []

    for C in [0.01, 0.05, 0.0625, 0.1, 0.3, 0.5, 1]:
        n += 1
        print('Data loaded, begining SVM ' + str(n) + ' training.')
        classifier = svm.SVC(C=C)  # Karnowski used C=0.0625

        classifier.fit(train_data, train_labels)

        acc = classifier.score(test_data, test_target)
        accs.append(acc)
        print(acc)
        if acc == max(accs):
            with open('SVM_best_model.txt', 'wb') as f:
                pickle.dump(classifier, f)
                max_acc = acc
    print([0.01, 0.05, 0.0625, 0.1, 0.3, 0.5, 1])
    print(accs)
    print(max_acc)