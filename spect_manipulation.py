from scipy import misc
from audioDataAnalysis.Utils import get_final_image, get_labels, get_files
import matplotlib.pyplot as plt
from numpy.random import choice
from scipy import ndimage
import numpy as np

mypath = '../KaggleData/'
image_path = '../KaggleData/spectograms/'

files = get_files(image_path)
labels = get_labels(mypath + 'train.csv', format='dict')

inv_map = {}
for k, v in labels.items():
    inv_map[v] = inv_map.get(v, [])
    inv_map[v].append(k)


file1 = choice(inv_map['1'], 1)[0]
file2 = choice(inv_map['0'], 1)[0]
#train10335.aiff
print('File2 printed')

im1 = misc.imread(image_path+file1[:-5] + '.png', mode='I')
im2 = misc.imread(image_path+file2[:-5] + '.png', mode='I')
im1 = get_final_image(im1, size='original', gray=True)
im2 = get_final_image(im2, size='original', gray=True)


print(im1.shape, im1.dtype)
print(im2.shape, im2.dtype)

fig = plt.figure()
a = fig.add_subplot(1,2,1)
plt.imshow(im1, cmap=plt.cm.gray)
a.set_title('Whale')
a=fig.add_subplot(1,2,2)
plt.imshow(im2, cmap=plt.cm.gray)
a.set_title('No whale')
plt.show()

print('Edge detection')


sx = ndimage.sobel(im1, axis=0, mode='constant')
sy = ndimage.sobel(im1, axis=1, mode='constant')
sob1 = np.hypot(sx, sy)

sx = ndimage.sobel(im2, axis=0, mode='constant')
sy = ndimage.sobel(im2, axis=1, mode='constant')
sob2 = np.hypot(sx, sy)

fig = plt.figure()
a = fig.add_subplot(1,2,1)
plt.imshow(sob1)
a.set_title('Sobeled Whale')
a=fig.add_subplot(1,2,2)
plt.imshow(sob2)
a.set_title('Sobeled no whale')
plt.show()
