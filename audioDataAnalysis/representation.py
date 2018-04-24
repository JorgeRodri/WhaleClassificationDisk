import matplotlib.pyplot as plt
from scipy.io import wavfile
import pickle
from sklearn.decomposition import PCA
import numpy as np

from PIL import Image

def pca(X):
  """  Principal Component Analysis
    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance
    and mean."""

  # get dimensions
  num_data, dim = X.shape

  # center data
  mean_X = X.mean(axis=0)
  X = X - mean_X

  if dim>num_data:
    # PCA - compact trick used
    M = np.dot(X,X.T) # covariance matrix
    e,EV = np.linalg.eigh(M) # eigenvalues and eigenvectors
    tmp = np.dot(X.T,EV).T # this is the compact trick
    V = tmp[::-1]# reverse since last eigenvectors are the ones we want
    S = np.sqrt(e)[::-1]# reverse since eigenvalues are in increasing order
    for i in range(V.shape[1]):
      V[:,i] /= S
  else:
    # PCA - SVD used
    U,S,V = np.linalg.svd(X)
    V = V[:num_data] # only makes sense to return the first num_data

  # return the projection matrix, the variance and the mean
  return V, S, mean_X


def graph_spectrogram(wav_file,save_name):
    rate, data = get_wav_info(wav_file)
    nfft = 256  # Length of the windowing segments
    fs = 4    # Sampling frequency
    pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, FS=fs, noverlap=10)
    plt.axis('off')
    plt.savefig(save_name,
                dpi=100, # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0) # Spectrogram saved as a .png

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

def get_PCA():
    with open('features.txt', 'rb') as f:
        alex = pickle.load(f)
    with open('featuresVGG16.txt', 'rb') as f:
        vgg16 = pickle.load(f)
    with open('featuresVGG19.txt', 'rb') as f:
        vgg19 = pickle.load(f)
    with open('labels.txt', 'rb') as f:
        y = pickle.load(f)

    y=np.array(y)
    target_names = ['No whale', 'Right whale']

    plt.figure()
    colors = ['red', 'navy']
    lw = 2

    pca = PCA(n_components=2)
    alex_r = pca.fit(alex).transform(alex)
    # Percentage of variance explained for each components
    print('Alexnet feature: explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))
    plt.subplot(3,1,1)
    for color, i, target_name in zip(colors, ['0', '1'], target_names):
        plt.scatter(alex_r[y == i, 0], alex_r[y == i, 1], s=5, color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA Alexnet dataset')


    vgg16_r = pca.fit(vgg16).transform(vgg16)
    # Percentage of variance explained for each components
    print('VGG16 features: explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.subplot(3,1,2)
    for color, i, target_name in zip(colors, ['0', '1'], target_names):
        plt.scatter(vgg16_r[y == i, 0], vgg16_r[y == i, 1], s=5, color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA VGG16 dataset')


    vgg19_r = pca.fit(vgg19).transform(vgg19)
    # Percentage of variance explained for each components
    print('VGG19 features: explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.subplot(3,1,3)
    for color, i, target_name in zip(colors, ['0', '1'], target_names):
        plt.scatter(vgg19_r[y == i, 0], vgg19_r[y == i, 1], s=5, color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA VGG19 dataset')

    plt.show()


def get_PCA_image():
    y=np.array()

    y=np.array(y)
    target_names = ['No whale', 'Right whale']

    plt.figure()
    colors = ['red', 'navy']
    lw = 2

    pca = PCA(n_components=2)
    alex_r = pca.fit(alex).transform(alex)
    # Percentage of variance explained for each components
    print('Alexnet feature: explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))
    plt.subplot(3,1,1)
    for color, i, target_name in zip(colors, ['0', '1'], target_names):
        plt.scatter(alex_r[y == i, 0], alex_r[y == i, 1], s=5, color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA Alexnet dataset')


    vgg16_r = pca.fit(vgg16).transform(vgg16)
    # Percentage of variance explained for each components
    print('VGG16 features: explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.subplot(3,1,2)
    for color, i, target_name in zip(colors, ['0', '1'], target_names):
        plt.scatter(vgg16_r[y == i, 0], vgg16_r[y == i, 1], s=5, color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA VGG16 dataset')


    vgg19_r = pca.fit(vgg19).transform(vgg19)
    # Percentage of variance explained for each components
    print('VGG19 features: explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.subplot(3,1,3)
    for color, i, target_name in zip(colors, ['0', '1'], target_names):
        plt.scatter(vgg19_r[y == i, 0], vgg19_r[y == i, 1], s=5, color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA VGG19 dataset')

    plt.show()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def get_LDA():
    with open('features.txt', 'rb') as f:
        alex = pickle.load(f)
    with open('featuresVGG16.txt', 'rb') as f:
        vgg16 = pickle.load(f)
    with open('featuresVGG19.txt', 'rb') as f:
        vgg19 = pickle.load(f)
    with open('labels.txt', 'rb') as f:
        y = pickle.load(f)

    y=np.array(y)
    target_names = ['No whale', 'Right whale']

    plt.figure()
    colors = ['red', 'navy']
    lw = 2

    clf = LDA()
    alex_r = clf.fit(alex, y)
    # Percentage of variance explained for each components
    print('Alexnet feature: explained variance ratio (first two components): %s'
          % str(clf.explained_variance_ratio_))
    plt.subplot(3,1,1)
    for color, i, target_name in zip(colors, ['0', '1'], target_names):
        plt.scatter(alex_r[y == i, 0], alex_r[y == i, 1], s=5, color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA Alexnet dataset')


    vgg16_r = clf.fit(vgg16, y)
    # Percentage of variance explained for each components
    print('VGG16 features: explained variance ratio (first two components): %s'
          % str(clf.explained_variance_ratio_))

    plt.subplot(3,1,2)
    for color, i, target_name in zip(colors, ['0', '1'], target_names):
        plt.scatter(vgg16_r[y == i, 0], vgg16_r[y == i, 1], s=5, color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA VGG16 dataset')


    vgg19_r = clf.fit(vgg19, y)
    # Percentage of variance explained for each components
    print('VGG19 features: explained variance ratio (first two components): %s'
          % str(clf.explained_variance_ratio_))

    plt.subplot(3,1,3)
    for color, i, target_name in zip(colors, ['0', '1'], target_names):
        plt.scatter(vgg19_r[y == i, 0], vgg19_r[y == i, 1], s=5, color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA VGG19 dataset')

    plt.show()