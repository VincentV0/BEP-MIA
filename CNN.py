"""
This file contains the most important functions to run the CNN model.
From the command line, this file can be ran using:
    python CNN.py <parameter_file>
When no parameter file is defined, the default parameter file is used.
"""

import os
import sys
import shutil

# Path where parameter files are stored are appended to the system path variable
sys.path.append('./parameters/')

# Default modules
import numpy as np
from importlib import import_module
from datetime import datetime

# Data processing modules
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import xlsxwriter

# TensorFlow modules
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Activation, concatenate, Conv2D, \
    MaxPooling2D, Conv2DTranspose, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, \
    TensorBoard, ReduceLROnPlateau
from tensorflow.keras import backend as K

# Metric scores
from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score,\
    recall_score, precision_score, matthews_corrcoef, confusion_matrix

# Loading functions from subfiles
from data import load_data, load_cleared_data
from data_augmentation import augment_data
from CNN_utils import *

# Information about the data, used by TensorFlow
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

################################################################################

def preprocess(imgs):
    """
    DESCRIPTION:
    -----------
    Preprocess data (downsampling and adding an axis)

    Parameters
    ----------
    imgs : TYPE np.ndarray
        Array with all the images.

    Returns
    -------
    imgs_p : TYPE np.ndarray
        Array with all the images, after the preprocessing step.

    """
    # Create an empty array to fill with the downsampled images.
    imgs_p = np.ndarray((imgs.shape[0], pm.img_rows_ds, pm.img_cols_ds), dtype=np.uint8)

    # Downsample the images using the resize function.
    imgs_p = resize(imgs, (imgs.shape[0], pm.img_rows_ds, pm.img_cols_ds), preserve_range=True)

    # Add an axis to the end of the array. This axis is required by TensorFlow.
    imgs_p = imgs_p[..., np.newaxis]

    # Return the preprocessed array.
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
        runnr = totalrunnr % pm.nb_folds;
        for i in range(len(pm.history_list[totalrunnr]['val_loss'])):
            worksheet.write(line, 0, 'FOLD {} RUN {}'.format(foldnr+1, runnr+1))
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
    worksheet.write(7, 7, 'MCC')
    worksheet.write(8, 7, 'TP')
    worksheet.write(9, 7, 'TN')
    worksheet.write(10, 7, 'FP')
    worksheet.write(11, 7, 'FN')

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
            worksheet.write(7, column, pm.MCC[foldnr*pm.runNum + runnr])
            worksheet.write(8, column, pm.TrPos[foldnr*pm.runNum + runnr])
            worksheet.write(9, column, pm.TrNeg[foldnr*pm.runNum + runnr])
            worksheet.write(10, column, pm.FaPos[foldnr*pm.runNum + runnr])
            worksheet.write(11, column, pm.FaNeg[foldnr*pm.runNum + runnr])
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
    # Model design is created
    print('-'*30)
    print('Creating model...')
    print('-'*30)

    # Model initialization
    model = keras.Sequential()

    # Model segment 1
    model.add(Conv2D(32, pm.conv_kernel_1, activation='relu', padding='same', input_shape = pm.input_shape_ds))
    model.add(Conv2D(32, pm.conv_kernel_1, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=pm.maxpool_kernel))

    # Model segment 2
    model.add(Conv2D(64, pm.conv_kernel_2, activation='relu', padding='same'))
    model.add(Conv2D(64, pm.conv_kernel_2, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=pm.maxpool_kernel))

    # Model segment 3
    model.add(Conv2D(128, pm.conv_kernel_3, activation='relu', padding='same'))
    model.add(Conv2D(128, pm.conv_kernel_3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=pm.maxpool_kernel))

    # Model segment 4
    model.add(Conv2D(256, pm.conv_kernel_4, activation='relu', padding='same'))
    model.add(Conv2D(256, pm.conv_kernel_4, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=pm.maxpool_kernel))

    # Model final segment (class determination)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(8, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(pm.dropout))
    model.add(Dense(pm.NB_CLASSES))
    model.add(Activation('softmax'))

    # Add the TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir="../tb_logs/"+pm.filename_run)

    # Another callback is used, which reduces the learning rate when a plateau
    # is reached on the monitored variable.
    reduce_lr_plateau = ReduceLROnPlateau(
        monitor = pm.RLRP_monitor,
        factor = pm.RLRP_factor,
        patience = pm.RLRP_patience,
        min_lr = pm.RLRP_minlr,
    )

    # Selecting optimizer with variable learning rate, and compile and fit the
    # model using these settings
    for ep in range(len(pm.nb_epochs)):
        optimizer=pm.model_optim(lr=pm.learning_rate[ep])
        model.compile(loss=pm.loss_function, optimizer=optimizer, metrics=pm.model_metrics)
        history = model.fit(trainingFeatures, trainingLabels, batch_size=pm.train_batch_size, \
            epochs=pm.nb_epochs[ep], verbose=pm.verbose_mode, shuffle=True, \
            validation_data=(valFeatures,valLabels),
            callbacks=[tensorboard_callback,reduce_lr_plateau])

        # Save the history for this set of epochs
        if ep == 0:
            hist_temp = history.history;
        else:
            for key in hist_temp:
                for item in history.history[key]:
                    hist_temp[key].append(item)

    # Saving the history of this run to the main list.
    pm.history_list.append(hist_temp)

    # Predict classes
    predicted_testLabels = model.predict_classes(testFeatures,verbose = 0)
    soft_targets_test = model.predict(testFeatures,verbose = 0)

    # Model prediction
    print('-'*30)
    print('Calculating scores...')
    print('-'*30)

    # Calculate scores and add to their list
    pm.precisionNet.append(precision_score(testLabels, predicted_testLabels))
    pm.recallNet.append(recall_score(testLabels, predicted_testLabels))
    pm.accNet.append(accuracy_score(testLabels, predicted_testLabels))
    pm.f1Net.append(f1_score(testLabels, predicted_testLabels))
    pm.AUCNet.append(roc_auc_score(testLabels, soft_targets_test[:,1]))
    tn, fp, fn, tp = confusion_matrix(testLabels, predicted_testLabels).ravel()
    pm.TrPos.append(tp)
    pm.TrNeg.append(tn)
    pm.FaPos.append(fp)
    pm.FaNeg.append(fn)
    pm.MCC.append(matthews_corrcoef(testLabels, predicted_testLabels))

    # Used for plotting the ROC curve
    fpr, tpr, _ = roc_curve(testLabels, soft_targets_test[:, 1])
    pm.fpr_list.append(fpr)
    pm.tpr_list.append(tpr)


################################################################################

if __name__ == '__main__':
    # Check whether a custom parameter file has been defined, and import this
    # parameter file. If none is defined, the default parameter file is used.
    if len(sys.argv) == 1:
        pm = import_module("params_default")

    elif len(sys.argv) == 2:
        # Try to load the custom parameter file, otherwise return an error.
        try:
            pm = import_module(sys.argv[1])
        except:
            print("Could not load parameter file '{}'.".format(sys.argv[1]))
            sys.exit(1)

    else:
        # If too many arguments are entered.
        print("Only one argument is allowed to define the parameter file.")
        sys.exit(2)

    # Initiate TensorBoard Log Dictionary
    try:
        os.mkdir(pm.tb_logs_path + pm.filename_run)
    except FileExistsError:
        q = input('TensorBoard folder already exists, remove or append to this? [R/A] ')
        if q.upper() == "R":
            shutil.rmtree(pm.tb_logs_path + pm.filename_run)
            os.mkdir(pm.tb_logs_path + pm.filename_run)
        elif q.upper() == "A": pass;
        else:
            print("Invalid option. Quitting.")
            exit()
    except:
        print("An unknown error happened. Quitting")
        exit()

    # Start the execution of the script here: start loading and preprocessing
    # data
    print('-'*30)
    print('Loading and preprocessing data...')
    print('-'*30)

    # Loading, preprocessing and splitting data
    data, labels = load_cleared_data()
    labels = make_labels(labels)
    data = preprocess(data)

    # Split the data using K-Fold cross validation
    print('-'*30)
    print('Splitting the data using K-Fold cross validation...')
    print('-'*30)
    KF_indices = split_test_data(data, pm.nb_folds)

    # Run this for every fold
    for fold in range(pm.nb_folds):
        print('-'*30)
        print('Assigning data for this fold...')
        print('-'*30)

        train_i, test_i = KF_indices[fold]

        # Return split datasets
        X_test  = data[test_i];
        X_train = data[train_i];
        y_test  = labels[test_i];
        y_train = labels[train_i];

        # Make multiple runs per fold
        for run in range(pm.runNum):
            # Start time is logged, to determine how long a run takes.
            start_time = datetime.now()

            # Data is normalized
            print('-'*30)
            print('Normalizing data...')
            print('-'*30)
            trainingFeatures, trainingLabels, testFeatures, testLabels = \
                normalization(X_train, y_train, X_test, y_test)

            # Labels are made categorical
            print('-'*30)
            print('Making labels categorical...')
            print('-'*30)
            trainingLabels = keras.utils.to_categorical(trainingLabels, pm.NB_CLASSES)

            # Using hold out validation to select validation set
            print('-'*30)
            print('Selecting and assigning validation set...')
            print('-'*30)
            train_indices,val_indices = train_test_split(np.arange(trainingFeatures.shape[0]),\
                test_size=pm.validation_fraction)
            valFeatures      = trainingFeatures[val_indices]
            trainingFeatures = trainingFeatures[train_indices]
            valLabels        = trainingLabels[val_indices]
            trainingLabels   = trainingLabels[train_indices]

            # Augmenting the training data and adding this to the training data set
            if pm.data_augm:
                print('-'*30)
                print('Augmenting data...')
                print('-'*30)
                augm_trainingFeatures, augm_trainingLabels = \
                    augment_data(trainingFeatures, trainingLabels, pm.nb_augm_samples, pm.augm_transformations)
                trainingFeatures = np.concatenate((trainingFeatures, augm_trainingFeatures), axis=0)
                trainingLabels = np.concatenate((trainingLabels, augm_trainingLabels), axis=0)

            # Run the main function
            train_and_predict()

            # Calculating and saving run time
            end_time = datetime.now()
            total_time = time_diff_format(start_time, end_time)
            pm.time_list.append(total_time)

    # Save the data to an xlsx-file and an image.
    write_save_data()
