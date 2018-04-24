from os import walk
import time
from scipy import misc
from audioDataAnalysis.Utils import get_final_image
import matplotlib.pyplot as plt
# from audioDataAnalysis.AlexNet import get_alexnet, AlexNet, get_alexnet2
#from keras.models import Model
from scipy.misc import imresize


image_path = '../KaggleData/spectograms/'
save_path = '../KaggleData/preprocessed/'

f = []
for (dirpath, dirnames, filenames) in walk(image_path):
    f.extend(filenames)
    break

t1 = time.time()
im = misc.imread(image_path+f[1])
final_im = get_final_image(im, size=[227, 227, 3])


# trial = im[6: 374, 33: 528, :3]
# final = imresize(trial, [224, 224, 3])
# print(trial.shape, trial.dtype)
# print(final.shape, final.dtype)
# plt.imshow(final)
# plt.show()

# data = 0
#
# input_size = (224, 224, 3)
# mean_flag = False
# batch_size = 16
# nb_classes = 10
#
# # alexnet = get_alexnet(input_size,nb_classes,mean_flag)
# # alexnet.load_weights('../weights/alexnet_weights.h5', by_name=True)
#
# from audioDataAnalysis.convnetkeras import convnet
# alexnet=convnet('alexnet', weights_path='../weights/alexnet_weights.h5')
#
# #alexnet_convolutional_only = Model(input=alexnet.input, output=alexnet.get_layer('convpool_5').output)
# alexnet_features = Model(input=alexnet.input, output=alexnet.get_layer('dense_2').output)

print('Image saved, importing keras packages.')
from keras.models import Model
from audioDataAnalysis.AlexNet import get_alexnet, AlexNet, get_alexnet2
from audioDataAnalysis.convnetkeras import convnet

print('Getting AlexNet. Oo')
alexnet = convnet('alexnet', '../weights/alexnet_weights.h5')
print('Loading weights...')
# alexnet.load_weights('../weights/alexnet_weights.h5', by_name=True)
alexnet_features = Model(input=alexnet.input, output=alexnet.get_layer('dense_2').output)

print('Weights loaded, lets predict!')

data = alexnet_features.predict(final_im, 93)
print(data)
