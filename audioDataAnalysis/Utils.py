from os import walk
from scipy.misc import imresize, imread, imsave
import time
import numpy as np
import random
from Lib import csv
import os, shutil
from math import floor
from random import randint


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def get_labels(file, format='dict'):
    if format=='dict':
        labels = dict()
        with open(file, 'r') as f:
            reader = csv.reader(f, dialect='excel')
            for row in reader:
                labels[row[0]] = row[1]
    elif format=='array':
        labels = list()
        with open(file, 'r') as f:
            reader = csv.reader(f, dialect='excel')
            for row in reader:
                labels.append(row)
    else:
        print('Wrong Format', 'Choose between: array and dict')
        labels=None
    return labels


def get_files(folder_path):
    audio_files = []
    for (dirpath, dirnames, filenames) in walk(folder_path):
        audio_files.extend(filenames)
        return audio_files


def get_images_ready_alexnet_batch(files, path, size):
    img_list = []
    for name in files:
        im = imread(path + name, mode='RGB')
        final = get_final_image(im, size=size)
        img = final.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        img = img.transpose((2, 0, 1))
        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                         ' in image_paths must have the same shapes.')
    return img_batch


def get_final_image(im, crop=(6, 374, 33, 528), size=(224, 224, 3), order='Tensorflow', gray=False):
    if gray:
        final_im = im[crop[0]: crop[1], crop[2]: crop[3]]
        if size == 'original':
            return final_im
        final = imresize(final_im, size[0:2])
    else:
        final_im = im[crop[0]: crop[1], crop[2]: crop[3], :3]
        if size == 'original':
            return final_im
        final = imresize(final_im, size)
    if order=='Theano':
        final = final.transpose((2, 0, 1))
        print(final.shape)
    return final


def cut_translate_rand(im, p=0.7):
    selection_range = floor(im.shape[1]*p)
    x_1 = randint(selection_range, im.shape[1]-1)
    x_0 = x_1-selection_range
    return im[:, x_0:x_1, :]


def cut_translate(im, p=0.8):
    selection_range = floor(im.shape[1]*p)
    x_1 = selection_range
    x_0 = x_1-selection_range
    return im[:, x_0:, :], im[:, 0:x_1, :]


def get_full_final(path, save_path, scale=(227, 227, 3), order='Tensorflow'):
    files = get_files(path)
    size = len(files)
    n = 0
    print('Starting, done: ' + str(n) + '%')
    times = [time.time()]
    for name in files:
        n += 1
        im = imread(path + name)
        final = get_final_image(im, size=scale, order=order)
        imsave(save_path + name, final)
        if n*100/size % 5 == 0:
            times.append(time.time())
            print('Done for: ' + str(n*100/size) + '%')
            print('Time for the last 5%: ' + str(times[-1] - times[-2]))
            print('Expected time to finish: ' + str((20-n*20/size) * (times[-1] - times[-2])/60) + ' minutes.')
    print('Done.')
    return 0


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def noisify(image, path, labels, _crop, _s, gray):
    files = os.listdir(path)
    boo = True
    while boo:
        random_file = np.random.choice(files, 1)[0]
        # print(random_file)
        if labels[random_file[:-4] + '.aiff'] == '0':
            im = imread(path + random_file)
            rand = get_final_image(im, crop=_crop, size= _s)
            if gray:
                rand = rgb2gray(rand)
            image += 0.28*rand
            image /= 1.28
            boo = False
    return image


# def get_full_final_labeled(path, save_path, label, test_value= 0.2, seed=60, _s=[224, 224, 3], _crop=[6, 374, 33, 528], gray=False):
#     files = get_files(path)
#     random.seed(seed)
#     random.shuffle(files)
#     size = len(files)
#     train = int(size*(1-test_value)//1)
#     n = 0
#     print('Starting, done: ' + str(n) + '%')
#     times = [time.time()]
#     for name in files:
#         n += 1
#         im = imread(path + name)
#         final = get_final_image(im, crop= _crop, size= _s)
#         if gray:
#             #get gray image
#             final = rgb2gray(final)
#         if n <= train:
#             if label[name[:-4]+'.aiff'] == '1':
#                 imsave(save_path + 'Train/' + 'whale/' + name, final)
#             elif label[name[:-4]+'.aiff'] == '0':
#                 imsave(save_path + 'Train/' + 'no_whale/' + name, final)
#         elif n > train:
#             if label[name[:-4] + '.aiff'] == '1':
#                 imsave(save_path + 'Test/' + 'whale/' + name, final)
#             elif label[name[:-4] + '.aiff'] == '0':
#                 imsave(save_path + 'Test/' + 'no_whale/' + name, final)
#         if n*100/size % 5 == 0:
#             times.append(time.time())
#             print('Done for: ' + str(n*100/size) + '%')
#             print('Time for the last 5%: ' + str(times[-1] - times[-2]))
#             print('Expected time to finish: ' + str((20-n*20/size) * (times[-1] - times[-2])//60) + ' minutes '+ str((20-n*20/size) * (times[-1] - times[-2])%60//1)+' seconds')
#     print('Done.')
#     return 0

def get_full_final_labeled(path, save_path, label, test_value=0.2, validation_split=0.0,  seed=60, _s=(224, 224, 3), _crop=(6, 374, 33, 528), gray=False):
    files = get_files(path)
    random.seed(seed)
    random.shuffle(files)
    size = len(files)
    train = int(size*(1-test_value-validation_split)//1)
    test = int(size*(1-validation_split)//1)
    n = 0
    print('Starting, done: ' + str(n) + '%')
    times = [time.time()]
    for name in files:
        n += 1
        im = imread(path + name)
        final = get_final_image(im, crop= _crop, size= _s)
        if gray:
            #get gray image
            final = rgb2gray(final)
        if n <= train:
            if label[name[:-4]+'.aiff'] == '1':
                imsave(save_path + 'Train/' + 'whale/' + name, final)
            elif label[name[:-4]+'.aiff'] == '0':
                imsave(save_path + 'Train/' + 'no_whale/' + name, final)
        elif n > train and n<=test:
            if label[name[:-4] + '.aiff'] == '1':
                imsave(save_path + 'Test/' + 'whale/' + name, final)
            elif label[name[:-4] + '.aiff'] == '0':
                imsave(save_path + 'Test/' + 'no_whale/' + name, final)
        elif n > test and validation_split != 0:
            if label[name[:-4] + '.aiff'] == '1':
                imsave(save_path + 'Validation/' + 'whale/' + name, final)
            elif label[name[:-4] + '.aiff'] == '0':
                imsave(save_path + 'Validation/' + 'no_whale/' + name, final)
        if n*100/size % 5 == 0:
            times.append(time.time())
            print('Done for: ' + str(n*100/size) + '%')
            print('Time for the last 5%: ' + str(times[-1] - times[-2]))
            print('Expected time to finish: ' + str((20-n*20/size) * (times[-1] - times[-2])//60) + ' minutes '+ str((20-n*20/size) * (times[-1] - times[-2])%60//1)+' seconds')
    print('Done.')
    return 0


def delete_files(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                delete_files(file_path)
        except Exception as e:
            print(e)


def get_full_final_enhanced(path, save_path, label, noisy_copies=2, test_value=0.2, validation_split=0.0,  seed=60, _s=(224, 224, 3), _crop=(6, 374, 33, 528), gray=False):
    files = get_files(path)
    random.seed(seed)
    random.shuffle(files)
    size = len(files)
    train = int(size*(1-test_value-validation_split)//1)
    test = int(size*(1-validation_split)//1)
    n = 0

    print('Deleting files before saving.')
    t1 = time.time()
    delete_files(save_path)
    print('Files deleted, time needed: %.2f.' % (time.time()-t1))

    print('Starting, done: ' + str(n) + '%')
    times = [time.time()]
    for name in files:
        n += 1
        im = imread(path + name)
        im2, im3 = cut_translate(im, 0.8)
        final = get_final_image(im, crop=_crop, size=_s)
        final2 = get_final_image(im2, crop=_crop, size=_s)
        final3 = get_final_image(im3, crop=_crop, size=_s)
        if gray:
            #get gray image
            final = rgb2gray(final)
            final2 = rgb2gray(final2)
            final3 = rgb2gray(final3)
        if n <= train:
            if label[name[:-4]+'.aiff'] == '1':
                imsave(save_path + 'Train/' + 'whale/' + name, final)
                imsave(save_path + 'Train/' + 'whale/translate1_' + name, final2)
                imsave(save_path + 'Train/' + 'whale/translate2_' + name, final3)
                # imsave(save_path + 'Train/' + 'whale/noisy_translate_' + name, noisify(final2, path, label, _crop, _s, gray))
                for _ in range(noisy_copies):
                    final2 = noisify(final, path, label, _crop, _s, gray)
                    imsave(save_path + 'Train/' + 'whale/noisy{}_'.format(_) + name, final2)
            elif label[name[:-4]+'.aiff'] == '0':
                imsave(save_path + 'Train/' + 'no_whale/' + name, final)
        elif n > train and n<=test:
            if label[name[:-4] + '.aiff'] == '1':
                imsave(save_path + 'Test/' + 'whale/' + name, final)
            elif label[name[:-4] + '.aiff'] == '0':
                imsave(save_path + 'Test/' + 'no_whale/' + name, final)
        elif n > test and validation_split != 0:
            if label[name[:-4] + '.aiff'] == '1':
                imsave(save_path + 'Validation/' + 'whale/' + name, final)
            elif label[name[:-4] + '.aiff'] == '0':
                imsave(save_path + 'Validation/' + 'no_whale/' + name, final)
        if n*100/size % 5 == 0:
            times.append(time.time())
            print('Done for: ' + str(n*100/size) + '%')
            print('Time for the last 5%: ' + str(times[-1] - times[-2]))
            print('Expected time to finish: ' + str((20-n*20/size) * (times[-1] - times[-2])//60) + ' minutes '+ str((20-n*20/size) * (times[-1] - times[-2])%60//1)+' seconds')
    print('Done.')
    return 0
