
from __future__ import print_function

import os
import sys

# Path where parameter files are stored are appended to the system path variable
sys.path.append('./parameters/')
sys.path.append('./random_parameters/')
from importlib import import_module
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve, roc_auc_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix, average_precision_score
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime
import xlsxwriter
import matplotlib.pyplot as plt

from data import load_data, load_cleared_data
from data_augmentation import augment_data

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
    # Make the figure
    fig,ax = plt.subplots()

    # Plot the ROC curves for the first fold only (figures become unclear when
    # more folds are plotted)
    for i in range(pm.runNum):
        ax.plot(fpr[i], tpr[i], marker = '.', label='Run {}'.format(i),alpha=0.3)

    # Style plot and save to file
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
    binary_targets[binary_targets != 0] = 1
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
def write_save_data():
    """
    DESCRIPTION:
    -----------
    Write the accuracies and losses after every epoch to an xlsx-file, and the
    evaluation scores on the test set to a seperate mat file.

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    """
    # Get the datetime for a overall filename
    filename_time = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    folder = pm.save_path + 'RUN {} at {}/'.format(pm.filename_run, filename_time);

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
    line = 1;
    for totalrunnr in range(len(pm.history_list)):
        foldnr = totalrunnr // pm.nb_folds;
        runnr  = totalrunnr - foldnr * pm.nb_folds;
        for i in range(len(pm.history_list[totalrunnr]['val_loss'])):
            worksheet.write(line, 0, 'FOLD {} RUN {}'.format(foldnr + 1, runnr + 1))
            worksheet.write(line, 1, i+1)
            worksheet.write(line, 2, pm.history_list[totalrunnr]['val_loss'][i])
            worksheet.write(line, 3, pm.history_list[totalrunnr]['val_accuracy'][i])
            worksheet.write(line, 4, pm.history_list[totalrunnr]['loss'][i])
            worksheet.write(line, 5, pm.history_list[totalrunnr]['accuracy'][i])
            line += 1;
        line += 1

    # Also write the scores for every run to this file
    worksheet.write(1, 7, 'precision')
    worksheet.write(2, 7, 'AUC')
    worksheet.write(3, 7, 'recall')
    worksheet.write(4, 7, 'f1')
    worksheet.write(5, 7, 'acc')
    worksheet.write(6, 7, 'time')
    column = 8;
    for foldnr in range(pm.nb_folds):
        for runnr in range(pm.runNum):
            worksheet.write(0, column, "FOLD {} RUN {}".format(foldnr+1, runnr+1))
            worksheet.write(1, column, pm.precisionNet[foldnr*pm.runNum + runnr])
            worksheet.write(2, column, pm.AUCNet[foldnr*pm.runNum + runnr])
            worksheet.write(3, column, pm.recallNet[foldnr*pm.runNum + runnr])
            worksheet.write(4, column, pm.f1Net[foldnr*pm.runNum + runnr])
            worksheet.write(5, column, pm.accNet[foldnr*pm.runNum + runnr])
            worksheet.write(6, column, pm.time_list[foldnr*pm.runNum + runnr])
            column += 1;
    workbook.close()


    # Also write the history of the final epoch for every run to a different file
    workbook = xlsxwriter.Workbook(folder + 'history_final_epoch ' + filename_time + '.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('B1', 'val_loss')
    worksheet.write('C1', 'val_acc')
    worksheet.write('D1', 'train_loss')
    worksheet.write('E1', 'train_acc')
    for foldnr in range(1,pm.nb_folds+1):
        for runnr in range(1,pm.runNum+1):
            line = (foldnr-1)*(pm.runNum+1) + runnr;
            worksheet.write(line, 0, 'FOLD {} RUN {}'.format(foldnr, runnr))
            worksheet.write(line, 1, pm.history_list[(foldnr-1)*pm.runNum + runnr-1]['val_loss'][-1])
            worksheet.write(line, 2, pm.history_list[(foldnr-1)*pm.runNum + runnr-1]['val_accuracy'][-1])
            worksheet.write(line, 3, pm.history_list[(foldnr-1)*pm.runNum + runnr-1]['loss'][-1])
            worksheet.write(line, 4, pm.history_list[(foldnr-1)*pm.runNum + runnr-1]['accuracy'][-1])
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
    start_time = datetime.now()   # Used to measure time taken

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
    model.add(Conv2D(32, pm.conv_kernel_1, activation='relu', padding='same', input_shape = pm.input_shape_ds))
    model.add(Conv2D(32, pm.conv_kernel_1, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=pm.maxpool_kernel))

    model.add(Conv2D(64, pm.conv_kernel_2, activation='relu', padding='same'))
    model.add(Conv2D(64, pm.conv_kernel_2, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=pm.maxpool_kernel))

    model.add(Conv2D(128, pm.conv_kernel_3, activation='relu', padding='same'))
    model.add(Conv2D(128, pm.conv_kernel_3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=pm.maxpool_kernel))

    model.add(Conv2D(256, pm.conv_kernel_4, activation='relu', padding='same'))
    model.add(Conv2D(256, pm.conv_kernel_4, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=pm.maxpool_kernel))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(8, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(pm.NB_CLASSES))
    model.add(Activation('softmax'))


    # Add the TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir="../tb_logs/"+pm.filename_run)
    earlystopping_callback = EarlyStopping(
        monitor=pm.es_monitor,      # Quantity to be monitored
        min_delta=pm.es_mindelta,   # Minimum change to qualify as 'improvement'
        patience=pm.es_patience,    # Number of epochs with no epochs, before training is stopped
        verbose=0,
        mode="auto",                # Automatically determined whether minimize or maximize the loss
        baseline=None,
        restore_best_weights=True,  # When the model is stopped, the model weights with the best monitored quantity are used.
    )
    reduce_lr_plateau = ReduceLROnPlateau(
        monitor = pm.RLRP_monitor,
        factor = pm.RLRP_factor,
        patience = pm.RLRP_patience,
        min_lr = pm.RLRP_minlr,
    )

    # Selecting optimizer with variable learning rate, and compile and fit the model using these settings
    for ep in range(len(pm.nb_epochs)):
        optimizer=pm.model_optim(lr=pm.learning_rate[ep])
        model.compile(loss=pm.loss_function, optimizer=optimizer, metrics=pm.model_metrics)
        history = model.fit(trainingFeatures, trainingLabels, batch_size=pm.train_batch_size, epochs=pm.nb_epochs[ep], \
            verbose=pm.verbose_mode, shuffle=True, validation_split=pm.validation_fraction, \
            callbacks=[tensorboard_callback,reduce_lr_plateau]) #class_weight = class_w

        # Save the history for this set of epochs
        if ep == 0:
            hist_temp = history.history;
        else:
            for key in hist_temp:
                for item in history.history[key]:
                    hist_temp[key].append(item)

    # Predict classes
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

    pm.history_list.append(hist_temp)

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

    # Initiate TensorBoard Log Dictionary
    try:
        os.mkdir('../tb_logs/' + pm.filename_run)
    except:
        pass;
    # Start the execution of the script here
    print('-'*30)
    print('Loading and preprocessing data...')
    print('-'*30)

    # Loading, preprocessing and splitting data
    data, labels = load_cleared_data()
    labels = make_labels(labels)
    data = preprocess(data)

    print('-'*30)
    print('Augmenting data...')
    print('-'*30)
    data, labels = augment_data(data, labels, pm.nb_samples)

    print('-'*30)
    print('Splitting the data using K-Fold cross validation...')
    print('-'*30)
    indices = split_data(data, labels, pm.nb_folds)

    # Run this for every fold
    for fold in range(pm.nb_folds):
        print('-'*30)
        print('Assigning data for this fold...')
        print('-'*30)

        train_i, test_i = indices[fold]

        # Return split datasets
        X_test  = data[test_i];
        X_train = data[train_i];
        y_test  = labels[test_i];
        y_train = labels[train_i];

        print(X_test)
        print(y_test)
        # Run the main loop.
        for run in range(pm.runNum):
            train_and_predict()

    # Save the data to an xlsx-file and an image.
    write_save_data()
