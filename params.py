# The paramters for the CNN are defined here.

# Import this to set the model optimizer
from tensorflow.keras.optimizers import Adam

# Constants
NB_CLASSES = 2
IMG_ROWS = 420
IMG_COLS = 580

# Downsampling of images
img_rows_ds = 96
img_cols_ds = 96
input_shape_ds = (img_rows_ds,img_cols_ds,1)

# Location of data
save_path = '/data/vrjvousten/results/';  # path to save the results to
data_path = '/data/vrjvousten/data/';    # path where the data is located

# Function of these variables is yet unknown
runNum = 5

# Function unknown
fpr = dict()
tpr = dict()
roc_auc = dict()

## Variables
#test_fraction = 0.2;        # fraction of data to be used for the test set. defaults to 0.2.
nb_folds = 5
validation_fraction = 0.2;  # fraction of the remaining train data to be used as validation

optimizer_learning_rate = 1e-5;
model_optimizer = Adam(lr=optimizer_learning_rate);
loss_function = 'binary_crossentropy';
model_metrics = ['accuracy'];

train_batch_size = 32;      # number of samples for every gradient update
nb_epochs = 20;          # number of epochs to train the model
verbose_mode = 1;           # verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
tensorflow_verbose = '1';  # TF verbosity mode (0 = print all information, 1 = hide info, 2 = hide info + warning,
                            # 3 = hide all)
