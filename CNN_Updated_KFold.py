

from __future__ import print_function

import os
import sys
from importlib import import_module
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve, roc_auc_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix, average_precision_score
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime
import xlsxwriter
import matplotlib.pyplot as plt


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

################### Simple functions ###########################################
def time_diff_format(start, end):
    """
    DESCRIPTION: This function calculates the time difference and returns it
    in a mm:ss format

    Parameters:
    ----------
    start: TYPE datetime
        the start-time
    end: TYPE datetime
        the end-time

    Returns
    -------
    time_formatted : TYPE string
        The time difference in mm:ss format

    """
    time_taken = (end - start).seconds;
    time_mins = time_taken // 60;
    time_secs = time_taken - (time_mins * 60);
    time_formatted = '{}:{}'.format(time_mins,time_secs)
    return time_formatted

################################################################################
def plot_ROC_curve(fpr, tpr, folder, filetime):
    """
    DESCRIPTION: This function plots the ROC curves of the different runs
    and saves it to a png file.

    Parameters:
    ----------
    fpr: the list with the false positive rates for every run
    tpr: the list with the true positive rates for every run
    folder: the folder to save to
    filetime: the time to be included in the filename.

    Returns
    -------
    time_formatted : TYPE string
        The time difference in mm:ss format

    """
    fig,ax = plt.subplots()
    for i in range(pm.runNum):
        ax.plot(fpr[i], tpr[i], marker = '.', label='Run {}'.format(i),alpha=0.3)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend()
    fig.savefig(folder + "ROC_curve " + filetime)

################################################################################
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


################################################################################
def split_data(imgs, targets, nb_folds):
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
    nb_folds : TYPE int, optional
        The value of k in the k-fold cross-validation algorithm. Default is 5.

    Returns
    -------
    indices : TYPE list
        List with arrays, which contain the training and test indices for all the
        runs
    """
    # Initiate K-Fold
    kf = KFold(n_splits=nb_folds,shuffle=True);
    # Get and return indices
    indices = list(kf.split(imgs))
    return indices


################################################################################
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
    imgs_p = np.ndarray((imgs.shape[0], pm.img_rows_ds, pm.img_cols_ds), dtype=np.uint8)
    imgs_p = resize(imgs, (imgs.shape[0], pm.img_rows_ds, pm.img_cols_ds), preserve_range=True)
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

################################################################################
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
    filename_time = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    folder = pm.save_path + 'RUN at ' + filename_time + '/';
    # Make a folder for all the results
    os.mkdir(folder)

    # Plot ROC curve and save to the folder
    plot_ROC_curve(pm.fpr_list, pm.tpr_list, folder, filename_time)

    # Save the loss/accuracy history to a xlsx file
    workbook = xlsxwriter.Workbook(folder + 'history ' + filename_time + '.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('B1', 'epoch')
    worksheet.write('C1', 'val_loss')
    worksheet.write('D1', 'val_acc')
    worksheet.write('E1', 'train_loss')
    worksheet.write('F1', 'train_acc')

    for runnr in range(1,pm.runNum+1):
        for i in range(pm.nb_epochs):
            line = (runnr-1)*pm.nb_epochs + i + 1 + runnr;
            worksheet.write(line, 0, 'RUN {}'.format(runnr))
            worksheet.write(line, 1, i+1)
            worksheet.write(line, 2, hist[runnr-1]['val_loss'][i])
            worksheet.write(line, 3, hist[runnr-1]['val_accuracy'][i])
            worksheet.write(line, 4, hist[runnr-1]['loss'][i])
            worksheet.write(line, 5, hist[runnr-1]['accuracy'][i])

    worksheet.write(1, 7, 'precision')
    worksheet.write(2, 7, 'AUC')
    worksheet.write(3, 7, 'recall')
    worksheet.write(4, 7, 'f1')
    worksheet.write(5, 7, 'acc')
    worksheet.write(6, 7, 'time')
    column = 8;
    for runnr in range(pm.runNum):
        worksheet.write(0, column, "RUN {}".format(runnr+1))
        worksheet.write(1, column, pm.precisionNet[runnr])
        worksheet.write(2, column, pm.AUCNet[runnr])
        worksheet.write(3, column, pm.recallNet[runnr])
        worksheet.write(4, column, pm.f1Net[runnr])
        worksheet.write(5, column, pm.accNet[runnr])
        worksheet.write(6, column, pm.time_list[runnr])
        column += 1;

    workbook.close()

############## Main function ###################################################
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

    print('-'*30)
    print('normalize data...')
    print('-'*30)
    trainingFeatures, trainingLabels, testFeatures, testLabels = normalization(X_train, y_train, X_test, y_test)

    print('-'*30)
    print('Make labels categorical...')
    print('-'*30)
    trainingLabels = keras.utils.to_categorical(trainingLabels, pm.NB_CLASSES)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)


    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape = pm.input_shape_ds))
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
    model.add(Dense(pm.NB_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss=pm.loss_function, optimizer=pm.model_optimizer, metrics=pm.model_metrics)
    history = model.fit(trainingFeatures, trainingLabels, batch_size=pm.train_batch_size, epochs=pm.nb_epochs, \
        verbose=pm.verbose_mode, shuffle=True, validation_split=pm.validation_fraction) #class_weight = class_w

    predicted_testLabels = model.predict_classes(testFeatures,verbose = 0)
    soft_targets_test = model.predict(testFeatures,verbose = 0)

    ############## model prediction and evaluation ##############
    print('-'*30)
    print('Calculating scores...')
    print('-'*30)

    # Calculate scores and add to their list
    pm.precisionNet.append(precision_score(testLabels, predicted_testLabels))
    pm.recallNet.append(recall_score(testLabels, predicted_testLabels))
    pm.accNet.append(accuracy_score(testLabels, predicted_testLabels))
    pm.f1Net.append(f1_score(testLabels, predicted_testLabels))
    pm.AUCNet.append(roc_auc_score(testLabels, soft_targets_test[:,1]))
    fpr, tpr, _ = roc_curve(testLabels, soft_targets_test[:, 1])

    pm.fpr_list.append(fpr)
    pm.tpr_list.append(tpr)

    end_time = datetime.now()
    total_time = time_diff_format(start_time, end_time)
    pm.time_list.append(total_time)

    pm.history_list.append(history.history)

################################################################################
if __name__ == '__main__':
    # Check whether a custom parameter file has been defined
    if len(sys.argv) == 1:
        pm = import_module("params_default")
    elif len(sys.argv) == 2:
        # Try to load the custom parameter file, otherwise return an error.
        try:
            pm = import_module(sys.argv[1])
        except:
            print("Could not find parameter file '{}'.".format(sys.argv[1]))
            sys.exit(1)
    else:
        # When too many arguments are given.
        print("Only one argument is allowed to define the parameter file.")
        sys.exit(2)

    # Start the execution of the script here
    print('-'*30)
    print('Loading and preprocessing data and applying K-Fold splitting...')
    print('-'*30)

    # Loading, preprocessing and splitting data
    data, labels = load_data()
    labels = make_labels(labels)
    data = preprocess(data)
    indices = split_data(data, labels, pm.runNum)

    for run in range(pm.runNum):
        start_time = datetime.now()   # Used to measure time taken
        print('-'*30)
        print('Assigning data for this run...')
        print('-'*30)
        # Assigning the data indices for this run
        train_i, test_i = indices[run]
        # Return split datasets
        X_test  = data[test_i];
        X_train = data[train_i];
        y_test  = labels[test_i];
        y_train = labels[train_i];
        train_and_predict()

    # Save the data to an xlsx-file and an image.
    write_save_data(pm.history_list)
