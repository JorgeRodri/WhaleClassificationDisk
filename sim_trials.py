from audioDataAnalysis.similarity import  dtw, fastdtw, _traceback
from audioDataAnalysis.Utils import get_files
import numpy
from Lib import aifc
from time import time

if __name__ == '__main__':
    # 1-D numeric

    mypath = '../KaggleData/train/'

    audio_files = get_files(mypath)

    from sklearn.metrics.pairwise import manhattan_distances
    with aifc.open(mypath+audio_files[1], 'r') as f:
        nframes = f.getnframes()
        strsig = f.readframes(nframes)
        x = numpy.fromstring(strsig, numpy.short).byteswap()
    with aifc.open(mypath+audio_files[3], 'r') as f:
        nframes = f.getnframes()
        strsig = f.readframes(nframes)
        y = numpy.fromstring(strsig, numpy.short).byteswap()
    dist_fun = manhattan_distances

    print('Files opened, calculating the dtw.')
    t1=time()
    dist, cost, acc, path = dtw(x, y, dist_fun)
    t2=time()
    print('Enlapsed time of: ' + str(t2-t1))

    # print('Files opened, calculating the fastdtw.')
    # t1 = time()
    # dist, cost, acc, path = fastdtw(x, y, dist_fun)
    # t2 = time()
    # print('Enlapsed time of: ' + str(t2 - t1))

    print('Starting Visualization')
    # vizualize
    from matplotlib import pyplot as plt
    plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
    plt.plot(path[0], path[1], '-o') # relation
    plt.xticks(range(len(x)), x)
    plt.yticks(range(len(y)), y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('tight')
    plt.title('Minimum distance: {}'.format(dist))
    plt.show()
