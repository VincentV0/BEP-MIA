"""
this script read all training and test data and masks.
prepare training data and masks and save them in a binary format.
prepare test data and test ids and save them in a binary format.

"""

from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from params_default import *

# Imports for removing duplicates
import scipy.spatial.distance as spdist
import skimage.util
from keras import utils
import glob
import cv2
import pylab as pl
import shutil
import matplotlib.cm as cm

################################################################################
def remove_duplicates(img_array, mask_array, img, mask, array_i):
    """
    DESCRIPTION:
    There are some duplicate images in the dataset, some of which have a mask and
    some of which don't. Double images WITHOUT MASK will be removed, if there is
    a mask, it will be kept.
    """
    pass

def dice(Y_pred, Y):
    """
    This works for one image
    http://stackoverflow.com/a/31275008/116067
    """
    denom = (np.sum(Y_pred == 1) + np.sum(Y == 1))
    if denom == 0:
        # By definition, see https://www.kaggle.com/c/ultrasound-nerve-segmentation/details/evaluation
        return 1
    else:
        return 2 * np.sum(Y[Y_pred == 1]) / float(denom)

def load_and_preprocess(imgname):
    img_fname = imgname
    mask_fname = os.path.splitext(imgname)[0] + "_mask.tif"
    img = cv2.imread(os.path.join(data_path, img_fname), cv2.IMREAD_GRAYSCALE) / 255
    mask = cv2.imread(os.path.join(data_path, mask_fname), cv2.IMREAD_GRAYSCALE) / 255
    return img, mask

def load_patient(pid):
    fnames = [os.path.basename(fname) for fname in glob.glob(data_path + "/%d_*.tif" % pid) if 'mask' not in fname]
    imgs, masks = zip(*map(load_and_preprocess, fnames))
    return imgs, masks, fnames

def compute_img_hist(img):
    # Divide the image in blocks and compute per-block histogram
    blocks = skimage.util.view_as_blocks(img, block_shape=(20, 20))
    img_hists = [np.histogram(block, bins=np.linspace(0, 1, 10))[0] for block in blocks]
    return np.concatenate(img_hists)

def filter_images_for_patient(pid):
    imgs, masks, fnames = load_patient(pid)
    print('Images for patient {}: {}'.format(pid, len(imgs)))
    #hists = np.array([np.histogram(img, bins=np.linspace(0, 1, 20))[0] for img in imgs]) ### CHECKEN
    hists = np.array(list(map(compute_img_hist, imgs)))
    D = spdist.squareform(spdist.pdist(hists, metric='cosine'))

    # Used 0.005 to train at 0.67
    close_pairs = D + np.eye(D.shape[0]) < 0.008

    close_ij = np.transpose(np.nonzero(close_pairs))

    incoherent_ij = [(i, j) for i, j in close_ij if dice(masks[i], masks[j]) < 0.2]
    incoherent_ij = np.array(incoherent_ij)

    #i, j = incoherent_ij[np.random.randint(incoherent_ij.shape[0])]

    valids = np.ones(len(imgs), dtype=np.bool)
    for i, j in incoherent_ij:
        if np.sum(masks[i]) == 0:
            valids[i] = False
        if np.sum(masks[j]) == 0:
            valids[i] = False

    item = 0;
    imgs_new = []
    masks_new = []
    for i in np.flatnonzero(valids):
        #imgname = os.path.splitext(fnames[i])[0] + ".png"
        #mask_fname = os.path.splitext(imgname)[0] + "_mask.png"
        #img = skimage.img_as_ubyte(imgs[i])
        #cv2.imwrite(os.path.join(OUTDIR, imgname), img)
        #mask = skimage.img_as_ubyte(masks[i])
        #cv2.imwrite(os.path.join(OUTDIR, mask_fname), mask)
        imgs_new.append(imgs[i])
        masks_new.append(masks[i])
        item += 1
    print('Discarded ', np.count_nonzero(~valids), " images for patient %d" % pid)
    return imgs_new, masks_new

################################################################################


def create_data():
    """

    DESCRIPTION:
    -------

    read data and masks, save them in seperate npy files
    """
    images = os.listdir(data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, IMG_ROWS, IMG_COLS), dtype=np.uint8)
    imgs_mask = np.ndarray((total, IMG_ROWS, IMG_COLS), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        try:
            img = imread(os.path.join(data_path, image_name), as_gray=True)
            img_mask = imread(os.path.join(data_path, image_mask_name), as_gray=True)
        except:
            print("Unknown file type found in file {} or {}: skipping".format(image_name, image_mask_name))
            continue

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    print('Removing duplicates using the histogram method.')
    # Method used from https://github.com/julienr/kaggle_uns/blob/master/13_clean/0_filter_incoherent_images.ipynb
    imgs_new = []
    masks_new = []
    for pid in range(1, 48):
        imgs_pat, imgs_mask_pat = filter_images_for_patient(pid)
        for itemID in range(len(imgs_pat)):
            imgs_new.append((imgs_pat[itemID]*255).astype(np.uint8))
            masks_new.append((imgs_mask_pat[itemID]*255).astype(np.uint8))

    imgs_new = np.array(imgs_new)
    masks_new = np.array(masks_new)

    np.save(os.path.join(data_path,'imgs1.npy'), imgs_new)
    np.save(os.path.join(data_path,'imgs1_mask.npy'), masks_new)
    print('Saving to .npy files done.')



def load_data():
    """
    load the saved masks and data

    Returns
    -------
    imgs_train : TYPE np.ndarray
        The images that are loaded from the npy-file.
    imgs_mask_train : TYPE
        The masks belonging with the images, loaded from the npy-file.

    """
    imgs_train = np.load(os.path.join(data_path,'imgs.npy'))
    imgs_mask_train = np.load(os.path.join(data_path,'imgs_mask.npy'))
    return imgs_train, imgs_mask_train


def load_cleared_data():
    """
    load the saved masks and data, but with the duplicates removed.

    Returns
    -------
    imgs_train : TYPE np.ndarray
        The images that are loaded from the npy-file.
    imgs_mask_train : TYPE np.ndarray
        The masks belonging with the images, loaded from the npy-file.

    """
    imgs_train = np.load(os.path.join(data_path,'data_clean/imgs.npy'))
    imgs_mask_train = np.load(os.path.join(data_path,'data_clean/imgs_mask.npy'))
    return imgs_train, imgs_mask_train

######################################################################
def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, IMG_ROWS, IMG_COLS), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

##########################################################################


if __name__ == '__main__':
    create_data()
    #create_test_data()
