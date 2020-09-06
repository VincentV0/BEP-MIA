

from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from data import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve, roc_auc_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix, average_precision_score
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime
from params import * # Load the parameters set in a different file

K.set_image_data_format('channels_last')  # TF dimension ordering in this code



def make_labels(targets):
    """
    DESCRIPTION
    -----------
    This function creates the labels for images based on the existance of BP:
    if the sum of values in the target is zeros returns '0' else returns '1'
    Parameters
    ----------
    targets : TYPE np.ndarray
        Contains the masks.

    Returns
    -------
    binary_targets : TYPE np.ndarray
        Contains the labels (0/1) of the images. If 1, the image has a non-empty
        mask. If 0, the image does have an empty mask.

    """
    targetShape = targets.shape;
    targetsReshaped = targets.reshape(targetShape[0],targetShape[1]*targetShape[2]);
    binary_targets = np.max(targetsReshaped,axis=1);
    binary_targets = binary_targets / 255;

    return binary_targets


###########################################

def split_data(imgs, targets, test_size = 0.2):
    """
    DESCRIPTION:
    -----------
    Split the data to a train and test set. This function uses k-fold cross
    validation.

    Parameters
    ----------
    imgs : TYPE np.ndarray
        An array with all the images loaded
    targets : TYPE np.ndarray
        An array with all the labels, corresponding to the images.
    test_size : TYPE np.ndarray, optional
        The fraction of the data that should be treated as test data. The rest
        of the data is used as training data. The default is 0.2.

    Returns
    -------
    X_train : TYPE np.ndarray
        Array with the training images.
    X_test : TYPE np.ndarray
        Array with the test images.
    y_train : TYPE np.ndarray
        Array with the training labels.
    y_test : TYPE np.ndarray
        Array with the test labels.

    """
    n_imgs = imgs.shape[0];
    k = int(np.floor(n_imgs * test_size));

    # Shuffle both image and mask datasets
    indices = np.arange(n_imgs);
    np.random.shuffle(indices);
    imgs = imgs[indices];
    targets = targets[indices];

    # Return split datasets
    X_test  = imgs[0:k];
    X_train = imgs[k:n_imgs];
    y_test  = targets[0:k];
    y_train = targets[k:n_imgs];

    return X_train,X_test,y_train,y_test


#############################################################################################################
def normalization(train_data, train_labels, test_data, test_labels):
    """
    DESCRIPTION:
    -----------
    Normalizes both the training data and test data, by substracting the training
    data mean and dividing by the training data standard deviation.

    Parameters
    ----------
    train_data : TYPE np.ndarray
        Array with the training images.
    train_labels : TYPE np.ndarray
        Array with the training labels.
    test_data : TYPE np.ndarray
        Array with the test images.
    test_labels : TYPE np.ndarray
        Array with the test labels.

    Returns
    -------
    train_data : TYPE np.ndarray
        Array with the training images.
    train_labels : TYPE np.ndarray
        Array with the training labels.
    test_data : TYPE np.ndarray
        Array with the test images.
    test_labels : TYPE np.ndarray
        Array with the test labels.

    """
    ###### data normalization ##########
    train_data = train_data.astype('float32')
    mean = np.mean(train_data)  # mean for data centering
    std = np.std(train_data)  # std for data normalization

    train_data -= mean
    train_data /= std

    test_data = test_data.astype('float32')
    test_data -= mean
    test_data /= std

    return train_data, train_labels, test_data, test_labels

########################################################
def preprocess(imgs):
    """
    DESCRIPTION:
    -----------
    Preprocess data via resizing images

    Parameters
    ----------
    imgs : TYPE np.ndarray
        Array with all the images.

    Returns
    -------
    imgs_p : TYPE np.ndarray
        Array with all the images, after the preprocessing step.

    """
    imgs_p = np.ndarray((imgs.shape[0], IMG_ROWS, IMG_COLS), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (IMG_ROWS, IMG_COLS), preserve_range=True) # change made here
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

################################################################################
def write_history(hist):
    """
    DESCRIPTION:
    -----------
    Write the accuracies and losses after every epoch to an xlsx-file.

    Parameters
    ----------
    hist : TYPE dictonary
        Constains the validation/training losses and accuracies for every epoch.

    Returns
    -------
    None.

    """
    workbook = xlsxwriter.Workbook(save_path + 'history ' + datetime.now().ctime() + '.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'epoch')
    worksheet.write('B1', 'val_loss')
    worksheet.write('C1', 'val_acc')
    worksheet.write('D1', 'train_loss')
    worksheet.write('E1', 'train_acc')

    for i in range(nb_epochs):
        worksheet.write(i+1, 0, i)
        worksheet.write(i+1, 1, hist['val_loss'][i])
        worksheet.write(i+1, 2, hist['val_sparse_categorical_accuracy'][i])
        worksheet.write(i+1, 3, hist['loss'][i])
        worksheet.write(i+1, 4, hist['sparse_categorical_accuracy'][i])

    workbook.close()

################################################################################
def train_and_predict():
    """
    DESCRIPTION:
    -----------
    This is the main function. The data is loaded and preprocessed, the model
    is created, compiled and fitted using the training data. Finally, the model
    is tested on the test data, after which post-processing steps are performed.

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    """
    start_time = datetime.now()   # Used to measure time taken

    print('-'*30)
    print('Loading and preprocessing data and spliting it to test and train...')
    print('-'*30)

    data, labels = load_data()
    labels = make_labels(labels)
    data = preprocess(data)

    X_train, X_test, y_train, y_test = split_data(data, labels, test_fraction)

    print('-'*30)
    print('normalize data...')
    print('-'*30)
    trainingFeatures, trainingLabels, testFeatures, testLabels = normalization(X_train, y_train, X_test, y_test)

    print('-'*30)
    print('Make labels categorical...')
    print('-'*30)

    trainingLabels = keras.utils.to_categorical(trainingLabels, NB_CLASSES)

    # Test data function is not used, but implemented in the split_data function
    # imgs_test, imgs_id_test = load_test_data()
    # imgs_test = preprocess(imgs_test)


    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)


    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape = INPUT_SHAPE))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(8, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))

    # model.summary()

    model.compile(loss=loss_function, optimizer=model_optimizer, metrics=model_metrics)
    history = model.fit(trainingFeatures, trainingLabels, batch_size=train_batch_size, epochs=nb_epochs, \
        verbose=verbose_mode, shuffle=True, validation_split=validation_fraction) #class_weight = class_w

    predicted_testLabels = model.predict_classes(testFeatures,verbose = 0)
    soft_targets_test = model.predict(testFeatures,verbose = 0)
    ############## model prediction and evaluation ##############

    print('-'*30)
    print('Calculating scores...')
    print('-'*30)

    precisionNet = precision_score(testLabels, predicted_testLabels)
    recallNet = recall_score(testLabels, predicted_testLabels)
    accNet = accuracy_score(testLabels, predicted_testLabels)
    f1Net = f1_score(testLabels, predicted_testLabels)
    print('precisionNet: %.4f' % (precisionNet))
    print('recallNet : %.4f'%(recallNet))

    AUCNet = roc_auc_score(testLabels, soft_targets_test[:,1])
    print('f1Net: %.4f' % (f1Net))
    print('AUCNet : %.4f'%(AUCNet))

    time_taken = (datetime.now() - start_time).seconds;
    time_mins = time_taken // 60;
    time_secs = time_taken - (time_mins * 60);
    time_formatted = '{}:{}'.format(time_mins,time_secs)


    sio.savemat(save_path + 'CNN_Results_' + datetime.now().ctime() + '.mat', \
        {'precisionNet': precisionNet,'AUCNet':AUCNet,
        'recallNet': recallNet, 'f1Net': f1Net,'accNet': accNet,
        'time': time_formatted})

    write_history(history.history)


if __name__ == '__main__':
    train_and_predict()
