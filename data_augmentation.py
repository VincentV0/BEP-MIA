"""
This script allows for data augmentation using different methods.

"""

import numpy as np
from scipy import ndimage


# SECTION 1. Geometrical transformations

def identity():
    "Returns a 2D-identity matrix"
    T = np.eye(2)
    return T


def scale(sx, sy):
    """
    Returns the transformation matrix that scales an image.
    Inputs are sx and sy (scale factors in x and y directions).
    """
    T = np.array([[sx,0],[0,sy]])

    return T


def rotate(phi):
    """
    Returns the transformation matrix that performs a rotation on an image.
    Input is phi; the angle (in rad) that the transformation matrix should produce.
    """
    T = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    return T


def shear(cx, cy):
    """
    Returns the transformation matrix that performs a shearing on an image.
    Inputs are cx and cy; the amount of shearing in x and y direction.
    """
    T = np.array([[1,cx],[cy,1]])

    return T


def reflect(rx, ry):
    """
    Returns the transformation matrix that performs a reflection on an image.
    Inputs are rx and ry (both should be either -1 or 1); this determines whether
    the image should be reflected in x and/or y direction.
    """
    allowed = [-1, 1]
    if rx not in allowed or ry not in allowed:
        T = 'Invalid input parameter'
        return T
    T = np.array([[rx,0],[0,ry]])
    return T


def t2h(T, t=np.array([0,0])):
    """
    Converts a 2D transformation matrix to homogeneous form. The default translation
    vector is [0,0], movement of the image is not included,
    """
    # Creates a row vector with zeroes with the size of T
    n = np.zeros([1,T.shape[1]])

    # Adds the row vector under the T matrix
    T1 = np.concatenate((T,n))

    # Adds a one under to the translation vector
    tn = np.append(t,1)

    # Creates the final homogeneous transformation matrix by adding the
    # translation vector as the final row to the T1 matrix
    Th = np.c_[T1,tn]
    return Th


def c2h(X):
    """
    Converts cartesian to homogeneous coordinates.
    """
    # Creates a row vector with ones with in the size of X.
    n = np.ones([1,X.shape[1]])

    # Creates the homogeneous coordinates by adding the vector of ones
    # under the matrix.
    Xh = np.concatenate((X,n))

    return Xh


def image_transform(I, Th):
    """
    This function transforms an image using the homogenous transformation matrix.
    """

    output_shape = I.shape

    # Spatial coordinates of the transformed image
    x = np.arange(0, output_shape[1])
    y = np.arange(0, output_shape[0])
    xx, yy = np.meshgrid(x, y)

    # Convert to a 2-by-p matrix (p is the number of pixels)
    X = np.concatenate((xx.reshape((1, xx.size)), yy.reshape((1, yy.size))))

    # Convert to homogeneous coordinates
    Xh = c2h(X)

    # Calculate the inverse of the homogenous transformation matrix.
    Th_inv =  np.linalg.inv(Th)

    # Calculate the dot product from the inverted Th with the homogenous
    # X-matrix.
    Xt = Th_inv.dot(Xh)

    # Calculate the transformed image.
    It = ndimage.interpolation.map_coordinates(I, [Xt[1,:], Xt[0,:]], order=1, mode='constant').reshape(I.shape)
    return It


def augment_data(data,labels,augm_nb_samples):
    """
    DESCRIPTION:
    -----------
    This function takes all the data and increases the number of samples by
    randomly selecting a transformation to be done.

    Returns:
    -------
    data: TYPE np.ndarray
        Array with images
    labels: TYPE nd.ndarray
        Array with the labels
    """
    # Record the number of samples before augmentation.
    nb_samples_start = data.shape[0]

    # Make two lists for augmented samples and labels
    augmented_data = []
    augmented_labels = []

    # Determine the possible values for the different transformations
    reflect_options = [-1, 1]
    scale_options = np.arange(1,2,0.05)
    rotate_options = np.arange(-np.pi, np.pi, 0.05*np.pi)
    shear_options = np.arange(-1,1,0.05)
    gaussian_options = np.arange(0.5, 5.5, 0.5)

    while len(augmented_data) <= augm_nb_samples:

        # Select image to be transformed:
        index = np.random.randint(nb_samples_start)
        im = data[index,:,:,0]
        im_label = labels[index]

        # Select random transformation:
        transformation_number = np.random.randint(5)
        if transformation_number == 0:
            # Reflection:
            rx, ry = 1, 1;
            while rx == 1 and ry == 1:
                rx = np.random.choice(reflect_options);
                ry = np.random.choice(reflect_options);
            T = reflect(rx,ry);

        if transformation_number == 1:
            # Scaling:
            sx = np.random.choice(scale_options);
            sy = np.random.choice(scale_options);
            T = scale(sx,sy);

        if transformation_number == 2:
            # Rotation:
            angle = np.random.choice(rotate_options);
            T = rotate(angle);

        if transformation_number == 3:
            # Shearing:
            cx = np.random.choice(shear_options);
            cy = np.random.choice(shear_options);
            T = shear(cx,cy);

        if transformation_number == 4:
            # Gaussian blur:
            sigma = np.random.choice(gaussian_options);
            im_T = ndimage.gaussian_filter(im, sigma=sigma);

        if transformation_number != 4:
            # Do some steps which are not required in Gaussian blurring
            # Check for singularity. When the matrix is singular, it cannot be
            # inverted or applied to the image and this step is reset.
            det = T[0,0]*T[1,1] - T[0,1]*T[1,0];
            if det == 0: continue;

            # Converts the 2D-transformation matrix to the homogenous form:
            Th = t2h(T);

            # Transform the image using the homogenous transformation matrix.
            im_T = image_transform(im,Th);

        # Add a new axis to be able to append to the data array
        im_T = im_T[..., np.newaxis];

        # Append data and labels to the augmented data lists
        augmented_data.append(im_T);
        augmented_labels.append(im_label);


    # Convert the lists to arrays
    print('{} samples augmented'.format(len(augmented_data)))
    augmented_data = np.array(augmented_data);
    augmented_labels = np.array(augmented_labels);

    # Remove duplicate augmented samples
    augmented_data_unique, i_unique = np.unique(augmented_data, return_index=True, axis=0);
    augmented_labels_unique = augmented_labels[i_unique];
    print('{} augmented samples left after removing duplicates'.format(augmented_data_unique.shape[0]))

    # Return augmented samples
    return augmented_data_unique, augmented_labels_unique
