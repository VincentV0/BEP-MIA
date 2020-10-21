"""
This file contains utility-functions that are required by the main function.
"""

# Import required functions
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

################################################################################

def dice_coef_numpy(y_true, y_pred, smooth=1.):
    """

    Parameters
    ----------
    y_true : TYPE
        the real labels.
    y_pred : TYPE
        the predicted labels via network.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    print('union = {}'.format(union))
    if union == 0: return 1.
    else: return (2. * intersection + smooth) / (union + smooth)

################################################################################
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

    # Plot the ROC curves for every run and every fold
    for i in range(len(fpr)):
        ax.plot(fpr[i], tpr[i], alpha=0.1, color='#ffaa75',LineWidth=.4)

    # Style plot and save to file
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    #ax.legend()
    fig.savefig(folder + "ROC_curve")

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

################################################################################

def split_test_data(imgs, nb_folds):
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
    # Determine original shape of the target array
    targetShape = targets.shape;

    # Reshape array, so that mask is 1D instead of 2D
    targetsReshaped = targets.reshape(targetShape[0],targetShape[1]*targetShape[2]);

    # Determine maximum value in masks; when this is not equal to zero, the image
    # has a mask and the label becomes 1. Else, the label becomes 0
    binary_targets = np.max(targetsReshaped,axis=1);
    binary_targets[binary_targets != 0] = 1

    # Return the labels
    return binary_targets
