from PIL import Image
from numpy import *
from pylab import *
from audioDataAnalysis.representation import pca
import os
from audioDataAnalysis.Utils import get_labels, get_files, get_final_image
import random

def get_imlist(path):
  """  Returns a list of filenames for
    all jpg images in a directory. """

  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

path = '../KaggleData/Spectrograms/spec_blackman/'
imlist = get_imlist(path)

labels_path = '../KaggleData/train.csv'
files_path = '../KaggleData/train/'
files = get_files(files_path)
labels = get_labels(labels_path)

whales = []
for i in files:
    if labels[i] == '1':
        whales.append(i)

random.shuffle(whales)

figure()
gray()
subplot(2, 5, 1)
n = 0
for i in whales[0:10]:
  n += 1
  subplot(2, 5, n)
  im = imread(path+i[:-5]+'.png')
  final = get_final_image(im, crop=[206, 374, 33, 528], size='original')
  imshow(final)
show()