from Lib import aifc
from os import walk
from matplotlib.mlab import window_hanning
import matplotlib.pyplot as plt
import numpy
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def get_spec_par(file):

    save_name='../KaggleData/spec_par/'

    with aifc.open(file, 'r') as f:
        nframes = f.getnframes()
        strsig = f.readframes(nframes)
        data = numpy.fromstring(strsig, numpy.short).byteswap()
    nfft = 256  # Length of the windowing segments
    fs = 2
    pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, noverlap=128)
    plt.axis('off')
    plt.savefig(save_name + file[20:-5] + '.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png


def get_spec_par_blackman(file):

    save_name='../KaggleData/spec_blackman/'

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

    save_name='../KaggleData/spec_hamming/'

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

    save_name='../KaggleData/spec_halfoverlap/'

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


def partitions(items, n):
    "Partitions the nodes into n subsets"
    nodes_iter = iter(items)
    while True:
        partition = tuple(itertools.islice(nodes_iter,n))
        if not partition:
            return
        yield partition


def show_spec(data, nfft=256, fs=2, noverlap=128, window=window_hanning):
    pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, noverlap=noverlap, window=window)
    plt.axis('off')
    return im


def get_spec(data, save_name):
    nfft = 256  # Length of the windowing segments
    fs = 2
    pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, noverlap=128)
    plt.axis('off')
    plt.savefig(save_name,
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png

def get_all_spec(datapath, savepath, proccesses=None):
    print('Starting...')
    files = []
    for (dirpath, dirnames, filenames) in walk(datapath):
        files.extend(filenames)
        break
    aux = []
    for (dirpath, dirnames, filenames) in walk(savepath):
        aux.extend(filenames)
        break
    done=[i[:-4] for i in aux]
    N=len(files)
    m=float(0)
    for file_name in files:
        if file_name[:-5] not in done:
            with aifc.open(datapath + file_name, 'r') as f:
                nframes = f.getnframes()
                strsig = f.readframes(nframes)
                data = numpy.fromstring(strsig, numpy.short).byteswap()
                get_spec(data, savepath + file_name[:-5] + '.png')
        done_for=m/N*100
        if done_for%5==0:
            print('Done for: '+str(done_for)+'%')
        m+=1
    print('done')


# To begin the parallel computation, we initialize a Pool object with the
# number of available processors on our hardware. We then partition the
# network based on the size of the Pool object (the size is equal to the
# number of available processors).
def get_par_spec(datapath, savepath, donepath):
    files = []
    for (dirpath, dirnames, filenames) in walk(datapath):
        files.extend(filenames)
        break
    done = []
    for (dirpath, dirnames, filenames) in walk(donepath):
        done.extend(filenames)
        break
    for (dirpath, dirnames, filenames) in walk(savepath):
        done.extend(filenames)
        break
    aux = [i[:-4] for i in done]
    pending = [datapath+i for i in files if i[:-5] not in aux]

    p = parallel_process(pending, get_spec_par)

    # p = Pool(processes=processes)
    # part_generator = 4 * len(p._pool)
    # node_partitions = list(partitions(pending, int(len(pending) / part_generator)))
    # num_partitions = int(len(node_partitions))
    #
    # # Next, we pass each processor a copy of the entire network and
    # # compute #the betweenness centrality for each vertex assigned to the
    # # processor.
    #
    # p.map(get_spec_par, pending)
    return p