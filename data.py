"""
this script read all training and test data and masks.
prepare training data and masks and save them in a binary format.
prepare test data and test ids and save them in a binary format.

"""

from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from params import *



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
        img = imread(os.path.join(data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(data_path, image_mask_name), as_gray=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(data_path,'imgs.npy'), imgs)
    np.save(os.path.join(data_path,'imgs_mask.npy'), imgs_mask)
    print('Saving to .npy files done.')


def load_data():
    """
    load the saved masks and data

    Returns
    -------
    imgs_train : TYPE
        DESCRIPTION.
    imgs_mask_train : TYPE
        DESCRIPTION.

    """
    imgs_train = np.load(os.path.join(data_path,'imgs.npy'))
    imgs_mask_train = np.load(os.path.join(data_path,'imgs_mask.npy'))
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

def write_save_data(hist):
    """
    DESCRIPTION:
    -----------
    Write the accuracies and losses after every epoch to an xlsx-file, and the
    evaluation scores on the test set to a seperate mat file.

    Parameters
    ----------
    hist : TYPE list
        Constains the validation/training losses and accuracies for every epoch.

    Returns
    -------
    None.

    """
    # Get the datetime for a overall filename
    filename_time = datetime.now().strftime("%Y_%m_%d %H_%M_%S")

    # Save the accuracy scores to a .mat file
    sio.savemat(save_path + 'CNN_Results_' + filename_time + '.mat', \
        {'precisionNet': precisionNet,'AUCNet':AUCNet,
        'recallNet': recallNet, 'f1Net': f1Net,'accNet': accNet,
        'fpr':fpr_list, 'tpr':tpr_list,
        'time': time_format_list})

    # Save the loss/accuracy history to a xlsx file
    workbook = xlsxwriter.Workbook(save_path + 'history ' + filename_time + '.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('B1', 'epoch')
    worksheet.write('C1', 'val_loss')
    worksheet.write('D1', 'val_acc')
    worksheet.write('E1', 'train_loss')
    worksheet.write('F1', 'train_acc')

    for runnr in range(1,runNum+1):
        for i in range(nb_epochs):
            line = (runnr-1)*nb_epochs + i + 1 + runnr;
            worksheet.write(line, 0, 'RUN {}'.format(runnr))
            worksheet.write(line, 1, i+1)
            worksheet.write(line, 2, hist[runnr-1]['val_loss'][i])
            worksheet.write(line, 3, hist[runnr-1]['val_accuracy'][i])
            worksheet.write(line, 4, hist[runnr-1]['loss'][i])
            worksheet.write(line, 5, hist[runnr-1]['accuracy'][i])

    workbook.close()

################################################################################


if __name__ == '__main__':
    create_data()
    #create_test_data()
