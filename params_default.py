"""
Parameter file
"""
# Import this to set the model optimizer
from tensorflow.keras.optimizers import Adam

# Constants
NB_CLASSES = 2;
IMG_ROWS = 420;
IMG_COLS = 580;

# Locations
save_path = '';    # path to save the results to
data_path = "E:\BEP_Data";       # path where the data is located
tb_logs_path = 'tb_logs'; # path to write the TensorBoard logs to

# Used for save data
fpr = dict();
tpr = dict();
roc_auc = dict();
precisionNet = [];
recallNet = [];
accNet = [];
f1Net = [];
AUCNet = [];
fpr_list = [];
tpr_list = [];
time_list = [];
history_list = [];
TrPos, TrNeg, FaPos, FaNeg = [],[],[],[];
MCC = [];

# TensorFlow verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
verbose_mode = 2;

# Downsampling of images
img_rows_ds = 96;
img_cols_ds = 96;
input_shape_ds = (img_rows_ds,img_cols_ds,1);

# General hyperparameters
filename_run  = 'default';  # Used to save the result data to a specific name
nb_folds      = 5;          # Number of folds in the K-Fold cross validation
runNum        = 10;         # Number of runs per fold
nb_epochs     = [25];       # Number of epochs per run
learning_rate = [1e-4];     # Starting learning rate (reduced by RLRP callback)
validation_fraction = 0.2;  # Fraction of the training data to be held out for validation
train_batch_size = 64;      # Number of samples used in the calculation for every gradient update

# Model fitting hyperparameters
model_optim = Adam;
loss_function = 'binary_crossentropy';
model_metrics = ['accuracy'];

# Model layer hyperparameters
conv_kernel_1 = (3, 3);     # Convolutional kernel size for 1st segment
conv_kernel_2 = (3, 3);     # Convolutional kernel size for 2nd segment
conv_kernel_3 = (3, 3);     # Convolutional kernel size for 3rd segment
conv_kernel_4 = (3, 3);     # Convolutional kernel size for 4th segment
maxpool_kernel = (2, 2);    # Max-pooling kernel size (all segments)
dropout       = 0.5;        # The parameter for the dropout layer

# "Reduce Learning Rate on Plateau" callback hyperparameters
RLRP_monitor = "val_accuracy";  # Variable to be monitored for a plateau
RLRP_patience = 2;              # Nr of epochs on a plateau before lowering the learning rate
RLRP_factor = 0.2;              # Factor to decrease learning rate by
RLRP_minlr = 1e-10;             # Lowest learning rate allowed

# Data augmentation hyperparameters
data_augm = True;             # Whether to apply data augmentation or not
nb_augm_samples = 5000;       # Total number of training samples after data augmentation
augm_transformations = ['reflect','scale','rotate','shear','gaussblur']
                              # Transformations to be applied to images
