import matplotlib.pyplot as plt
import pickle
import random
from scipy.misc import imread
from audioDataAnalysis.Utils import get_full_final_labeled, get_labels, get_files, get_full_final_enhanced
from audioDataAnalysis.representation import get_LDA
from scipy import misc

mypath = '../KaggleData/'
path = '~/Jorge-spectrograms/Spectrograms/spec_blackman'
labels_path = '~/Jorge-spectrograms/script/train.csv'
save_path = '~/Jorge-spectrograms/preprocessed_data'

if __name__ == '__main__':
    label = get_labels(labels_path)
    print('Doing it for ' + path)
    get_full_final_enhanced(path, save_path, label, noisy_copies=1,
                            validation_split=0.1, _crop=[230, 374, 33, 528], _s=[32, 32, 3], gray=True)
