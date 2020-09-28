# The paramters for the CNN are defined here.

# Import this to set the model optimizer
from tensorflow.keras.optimizers import Adam

# Constants
NB_CLASSES = 2;
IMG_ROWS = 420;
IMG_COLS = 580;

# Downsampling of images
img_rows_ds = 96;
img_cols_ds = 96;
input_shape_ds = (img_rows_ds,img_cols_ds,1);

# Location of data
save_path = '/data/vrjvousten/results/';  # path to save the results to
data_path = '/data/vrjvousten/data/';    # path where the data is located

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

verbose_mode = 1;           # verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).


## Hyperparameters:
filename_run = 'default';   # Used to save the result data to a specific name
nb_folds = 5;               # Number of folds to be used in k-fold cross validation
runNum = 10;                # nr of runs per fold
nb_epochs = [20];          # number of epochs to train the model
learning_rate = [1e-5];# variable learning rate for different amounts of epochs

validation_fraction = 0.2;  # fraction of the remaining train data to be used as
                            # validation (this uses hold-out validation)
train_batch_size = 32;      # number of samples used in the calculation for every gradient update

#model_optimizer = Adam(lr=learning_rate);
loss_function = 'binary_crossentropy';
model_metrics = ['accuracy'];

conv_kernel_1 = (3, 3);
conv_kernel_2 = (3, 3);
conv_kernel_3 = (3, 3);
conv_kernel_4 = (3, 3);
maxpool_kernel = (2, 2);

es_monitor = "val_loss";
es_mindelta = 0;   # any improvement is improvement
es_patience = 2;   # wait two epochs before exiting

# Total number of samples after augmenting data
nb_samples = 10000;
