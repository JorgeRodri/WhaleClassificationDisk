# from keras.optimizers import SGD
# from convnetskeras.convnets import preprocess_image_batch, convnet
# from keras import backend as K
#
# print(K.image_data_format())
# im = preprocess_image_batch(['examples/dog.jpg'],img_size=(256,256), crop_size=(227,227), color_mode="rgb")
#
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=False)
# model.compile(optimizer=sgd, loss='mse')
#
# out = model.predict(im)
# print(out)

# from keras.optimizers import SGD
# from convnetskeras.convnets import preprocess_image_batch, convnet
# from convnetskeras.imagenet_tool import synset_to_dfs_ids
# from keras.models import Model
#
# im = preprocess_image_batch(['examples/dog.jpg'],img_size=(256,256), crop_size=(227,227), color_mode="bgr")
#
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=False)
# model.compile(optimizer=sgd, loss='mse')
# alexnet_features = Model(input=model.input, output=model.get_layer('dense_2').output)
#
# print(im.shape)
#
# data = alexnet_features.predict(im)
#
# print(data)

from Lib import aifc
from audioDataAnalysis.kagglenet import show_spec
from audioDataAnalysis.Utils import get_files
import numpy
import matplotlib.pyplot as plt
from scipy import misc

mypath = '../KaggleData/train/'
files = get_files(mypath)
file_name = numpy.random.choice(files)

with aifc.open(mypath + file_name, 'r') as f:
    nframes = f.getnframes()
    strsig = f.readframes(nframes)
    data = numpy.fromstring(strsig, numpy.short).byteswap()
fig = plt.figure()
a = fig.add_subplot(2,2,1)
im = show_spec(data)
a.set_title('Original Spec')
a=fig.add_subplot(2,2,2)
im = show_spec(data, nfft=256, fs=2, noverlap=64)
a.set_title('Half overlap')
a = fig.add_subplot(2,2,3)
im = show_spec(data, nfft=256, fs=2, noverlap=128, window=numpy.blackman(256))
a.set_title('Blackman')
a=fig.add_subplot(2,2,4)
im = show_spec(data, nfft=256, fs=2, noverlap=128, window=numpy.hamming(256))
a.set_title('Hamming')
plt.show()
