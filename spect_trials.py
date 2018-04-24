
import numpy as np
from audioDataAnalysis.representation import *
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
from audioDataAnalysis.Utils import get_labels

path = '../2015DCLDEWorkshop/SocalLFDevelopmentData/CINMS17B_winter/'
path2 = '../KaggleData/train/'

labels = get_labels('../KaggleData/' + 'train.csv', format='dict')
whale_map = {'1': 'Whale', '0': 'No Whale'}

if __name__ == '__main__': # Main function
    # wav_file = path+'CINMS17B_d03_111202_012730.d100.x.wav' # Filename of the wav file
    for i in range(1, 11):
        aiff_file = path2 + 'train{}.aiff'.format(i)  # Filename of the wav file
        save_name = 'wholeWinter111202.png'
        y, sr = librosa.load(aiff_file, duration=2)
        ps = librosa.feature.melspectrogram(y=y, sr=sr, fmax = 1024)
        print(ps.shape)
        librosa.display.specshow(ps, y_axis='mel', x_axis='time')
        plt.title('Spectogram for train{0}.aiff, the label is {1}'.format(i, whale_map[labels['train{}.aiff'.format(i)]]))
        plt.savefig('C:/Users/jorge/Desktop/MAI/example{}.png'.format(i))
        plt.show()
        # time.sleep(5)
    # print(rate)
    # print(data)
    #graph_spectrogram(wav_file,save_name)

