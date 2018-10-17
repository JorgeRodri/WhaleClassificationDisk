import matplotlib.pyplot as plt
import aifc
from os import walk
import numpy
import os
import librosa
import librosa.display
import scipy.signal as signal


def get_spec_mel(file):
    save_name = '../Spectrograms/mel_spect/'
    save_name = 'C:\\Users\\jorge\\Desktop\\MAI\\'
    y, sr = librosa.load(file, duration=196)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, fmax=1024)
    librosa.display.specshow(ps, y_axis='mel', x_axis='time')
    plt.axis('off')
    plt.savefig(save_name + file[20:-5] + '.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png
    plt.show()
    plt.close()


def get_spec_scipy(file):
    save_name = '../Spectrograms/scipy_spect/'
    save_name = 'C:\\Users\\jorge\\Desktop\\MAI\\scipy'
    with aifc.open(file, 'r') as f:
        nframes = f.getnframes()
        strsig = f.readframes(nframes)
        data = numpy.fromstring(strsig, numpy.short).byteswap()
        f, t, sxx = signal.spectrogram(data)
        plt.pcolormesh(t, f, sxx)
    plt.axis('off')
    plt.savefig(save_name + file[20:-5] + '.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png
    plt.show()
    plt.close()


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def get_files(folder_path):
    audio_files = []
    for (dirpath, dirnames, filenames) in walk(folder_path):
        audio_files.extend(filenames)
        return audio_files


if __name__ == '__main__':
    path = '../KaggleData/train/'
    list_files = os.listdir(path)
    split = split_list(list_files, 8)
    print(len(split[0]))
    import random
    file = random.choice([path + fil for fil in list_files])
    get_spec_mel(file)

    get_spec_scipy(file)
