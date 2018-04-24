import aifc
import numpy
import matplotlib.pyplot as plt
from os import walk

def get_spec_par_blackman(file):
    save_name = '../Spectrograms/spec_blackman/'

    with aifc.open(file, 'r') as f:
        nframes = f.getnframes()
        strsig = f.readframes(nframes)
        data = numpy.fromstring(strsig, numpy.short).byteswap()
    nfft = 256  # Length of the windowing segments
    fs = 2
    pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, window=numpy.blackman(256))
    plt.axis('off')
    plt.savefig(save_name + file[20:-5] + '.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png


def get_spec_par_hamming(file):

    save_name='../Spectrograms/spec_hamming/'

    with aifc.open(file, 'r') as f:
        nframes = f.getnframes()
        strsig = f.readframes(nframes)
        data = numpy.fromstring(strsig, numpy.short).byteswap()
    nfft = 256  # Length of the windowing segments
    fs = 2
    pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, window=numpy.hamming(256))
    plt.axis('off')
    plt.savefig(save_name + file[20:-5] + '.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png


def get_spec_par_halfoverlap(file):

    save_name='../Spectrograms/spec_halfoverlap/'

    with aifc.open(file, 'r') as f:
        nframes = f.getnframes()
        strsig = f.readframes(nframes)
        data = numpy.fromstring(strsig, numpy.short).byteswap()
    nfft = 256  # Length of the windowing segments
    fs = 2
    pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, noverlap=64)
    plt.axis('off')
    plt.savefig(save_name + file[20:-5] + '.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def get_files(folder_path):
    audio_files = []
    for (dirpath, dirnames, filenames) in walk(folder_path):
        audio_files.extend(filenames)
        return audio_files

if __name__ == '__main__':
    path = '../KaggleData/train/'
    files = get_files(path)
    split = split_list(files, 8)
    print(len(split[0]))
