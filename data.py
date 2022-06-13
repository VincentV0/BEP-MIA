"""
this script read all training and test data and masks.
prepare training data and masks and save them in a binary format.
prepare test data and test ids and save them in a binary format.

"""

from __future__ import print_function

import os
import numpy as np

from skimage.io import imread
from params_default import *

# Imports for removing duplicates
import scipy.spatial.distance as spdist
import skimage.util
import glob
import cv2

################################################################################

def dice(Y_pred, Y):
    """
    DESCRIPTION:
    -----------
    Calculates the dice score of a predicted mask versus its ground truth.

    Parameters:
    ----------
    Y_pred: TYPE np.ndarray
        The predicted mask
    Y: TYPE np.ndarray
        The ground-truth mask

    Returns:
    --------
    dice: TYPE int
        The dice score of these masks
    """

    denom = (np.sum(Y_pred == 1) + np.sum(Y == 1))
    if denom == 0:
        # By definition, see https://www.kaggle.com/c/ultrasound-nerve-segmentation/details/evaluation
        return 1
    else:
        return 2 * np.sum(Y[Y_pred == 1]) / float(denom)

################################################################################

def load_and_preprocess(imgname):
    """
    DESCRIPTION:
    -----------
    This function loads the data and scales it between 0 and 1.

    Parameters:
    ----------
    imgname: TYPE string
        The filename of the image to be loaded.

    Returns:
    -------
    img: TYPE np.ndarray
        The image corresponding to the filename
    mask: TYPE np.ndarray
        The mask corresponding to the filename
    """

    img_fname = imgname
    mask_fname = os.path.splitext(imgname)[0] + "_mask.tif"
    img = cv2.imread(os.path.join(data_path, img_fname), cv2.IMREAD_GRAYSCALE) / 255
    mask = cv2.imread(os.path.join(data_path, mask_fname), cv2.IMREAD_GRAYSCALE) / 255
    return img, mask

################################################################################

def load_patient(pid):
    """
    DESCRIPTION:
    -----------
    This function loads the data from a specific patient.

    Parameters:
    ----------
    pid: TYPE int
        The patient ID

    Returns:
    -------
    imgs: TYPE list
        Contains all the images for this patient
    masks: TYPE list
        Contains all the masks for this patient
    fnames: TYPE list
        A list with all the filenames for images corresponding with this patient (no mask!)
    """

    fnames = [os.path.basename(fname) for fname in glob.glob(data_path + "/%d_*.tif" % pid) if 'mask' not in fname]
    imgs, masks = zip(*map(load_and_preprocess, fnames))
    return imgs, masks, fnames

################################################################################

def compute_img_hist(img):
    """
    DESCRIPTION:
    -----------
    This function computes the histogram of an image, after dividing the image
    in blocks of 20x20.

    Parameters:
    ----------
    img: TYPE np.ndarray
        The image of which the histogram should be calculated

    Returns:
    -------
    img_hists: TYPE np.ndarray
        The histogram of the image (when divided in blocks)
    """

    blocks = skimage.util.view_as_blocks(img, block_shape=(20, 20))
    img_hists = [np.histogram(block, bins=np.linspace(0, 1, 10))[0] for block in blocks]
    return np.concatenate(img_hists)

################################################################################

def filter_images_for_patient(pid):
    """
    DESCRIPTION:
    -----------
    This function filters the images for a specific patient, by comparing the
    histograms and looking for close matches with an incoherent mask.

    Parameters:
    ----------
    pid: TYPE int
        The patient ID

    Returns:
    -------
    imgs_new: TYPE list
        The list containing all the valid images
    masks_new: TYPE list
        The list containing all the valid masks
    """

    # Load patient data
    imgs, masks, fnames = load_patient(pid)
    print('Images for patient {}: {}'.format(pid, len(imgs)))

    # Compute histograms for patient and determine spatial distance between histograms
    hists = np.array(list(map(compute_img_hist, imgs)))
    D = spdist.squareform(spdist.pdist(hists, metric='cosine'))

    # Determine close pairs (excluding pairs of images with itself)
    close_pairs = D + np.eye(D.shape[0]) < 0.008
    close_ij = np.transpose(np.nonzero(close_pairs))

    # Determine the indices of close pairs that have an incoherent mask
    incoherent_ij = [(i, j) for i, j in close_ij if dice(masks[i], masks[j]) < 0.2]
    incoherent_ij = np.array(incoherent_ij)

    # Check whether the masks
    valids = np.ones(len(imgs), dtype=np.bool)
    for i, j in incoherent_ij:
        if np.sum(masks[i]) == 0:
            valids[i] = False
        if np.sum(masks[j]) == 0:
            valids[i] = False

    # Add all the valid images and masks to lists and return these
    item = 0;
    imgs_new = []
    masks_new = []
    for i in np.flatnonzero(valids):
        imgs_new.append(imgs[i])
        masks_new.append(masks[i])
        item += 1
    print('Discarded ', np.count_nonzero(~valids), " images for patient %d" % pid)
    return imgs_new, masks_new

################################################################################

def create_data():
    """
    DESCRIPTION:
    -----------
    Read data and masks, filter incoherent items and save them in seperate npy files.

    Parameters:
    ----------
    None

    Returns:
    -------
    None

    """

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    # Method used from https://github.com/julienr/kaggle_uns/blob/master/13_clean/0_filter_incoherent_images.ipynb
    # Make empty lists to store the data to
    imgs = []
    masks = []

    # There are 47 patients
    for pid in range(1, 48):
        # Load images of this specific patient and filter incoherent masks.
        imgs_pat, imgs_mask_pat, _ = load_patient(pid)

        # Add all images and masks to the list
        for itemID in range(len(imgs_pat)):
            imgs.append((imgs_pat[itemID]*255).astype(np.uint8))
            masks.append((imgs_mask_pat[itemID]*255).astype(np.uint8))

    # Convert the lists to arrays
    imgs = np.array(imgs)
    masks = np.array(masks)

    # Save the data to .npy files.
    np.save(os.path.join(data_path,'data_clean/imgs.npy'), imgs)
    np.save(os.path.join(data_path,'data_clean/imgs_mask.npy'), masks)
    print('Saving to .npy files done.')

################################################################################

def load_data(fn_imgs = 'imgs.npy', fn_masks = 'imgs_mask.npy'):
    """
    DESCRIPTION:
    ------------
    Load the saved masks and data

    Parameters:
    ----------
    None

    Returns:
    -------
    imgs_train : TYPE np.ndarray
        The images that are loaded from the npy-file.
    imgs_mask_train : TYPE np.ndarray
        The masks belonging with the images, loaded from the npy-file.
    """

    imgs_train = np.load(os.path.join(data_path,fn_imgs))
    imgs_mask_train = np.load(os.path.join(data_path,fn_masks))
    return imgs_train, imgs_mask_train

################################################################################

def load_cleared_data():
    """
    DESCRIPTION:
    -----------
    Load the saved masks and data, but with the duplicates removed.

    Parameters:
    ----------
    None

    Returns:
    -------
    imgs_train : TYPE np.ndarray
        The images that are loaded from the npy-file.
    imgs_mask_train : TYPE np.ndarray
        The masks belonging with the images, loaded from the npy-file.

    """

    imgs_train = np.load(os.path.join(data_path,'data_clean/imgs.npy'))
    imgs_mask_train = np.load(os.path.join(data_path,'data_clean/imgs_mask.npy'))
    return imgs_train, imgs_mask_train

################################################################################

def create_test_data():
    """
    DESCRIPTION:
    -----------
    Loads the test data from .tif files and saves this to a .npy file.

    Parameters:
    ----------
    None

    Returns:
    -------
    None
    """

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

################################################################################

def load_test_data():
    """
    DESCRIPTION:
    -----------
    Loads the test data from .npy files.

    Parameters:
    ----------
    None

    Returns:
    -------
    imgs_test: TYPE np.ndarray
        The test images

    imgs_id: TYPE np.ndarray
        The IDs corresponding to the test images
    """

    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

################################################################################


if __name__ == '__main__':
    create_data()
    #create_test_data()
