# Changelog
### 16 Oct 2020
- `CNN.py` updated, does no longer include datetime in filenames, only in result folder, and has some error catches built in;
- `CNN_utils.py` updated, does no longer include datetime in ROC-curve filename;
- Added `combi.py` and hyperparameter files, and `data_augmentation.py` was updated, for using both classification and segmentation models.

### 29 Sep 2020
- `CNN_utils.py` added, including functions that will be used by `CNN.py`
- `CNN_plateau_callback.py` has been renamed to `CNN.py`, which means that the original `CNN.py` has been removed. The EarlyStopping callback is not used anymore, but fully replaced by the Reduce LR on Plateau callback.
- `params_default.py` has been updated.
- `used_python_packages.txt` has been updated
- Updated docstrings and comments
- Updated README.md

### 28 Sep 2020
- Added the Reduce LR on Plateau callback.
- Updated the data loader to remove duplicate samples.
- Data is augmented using different methods (shearing, scaling, applying Gaussian blurring and more).
  - `data_augmentation.py` has been added.

### 20 Sep 2020
- Added the TensorBoard and EarlyStopping callbacks.
- Added a variable convolution kernel size for the different layer "segments".

### 12 Sep 2020
- Result data processing is now updated, but not yet tested.
- `CNN_Updated_KFold.py` renamed to `CNN.py`; old `CNN.py` has been removed.

### 11 Sep 2020
- Added `CNN_Updated_KFold.py`; this file should have the K-Fold cross validation applied correctly.
- Added custom folder name per parameter file (in `CNN_Updated_KFold.py` for now only)

### 10 Sep 2020
- It is now possible to push a specific parameter file as an argument through the command line.
- Problem in the preprocess function is now fixed.
- Added `plot_ROC_curve` function, which saves the ROC curve to a .png file.
- Changed name of `params.py` to `params_default.py`; the `data.py` file always takes the parameters from here.
- Updated docstrings

### 8 Sep 2020
- Update on `preprocess`function: downsampling does now work properly.
- Update in the `split_data` function: k-fold cross validation is now done by using the sklearn KFold function. The `test_fraction` variable has changed to `nb_folds`, as this is required by the KFold function.
- `write_history` has been changed to `write_save_data`; all the results are now written from here.
- The "false positive rate" and "false negative rate", as well as the "Area Under Curve" (AUC) for these curves, has been added to the "Results" .mat-file.
- Filenames of the output files do now have matching times, and the : are removed (because of errors with these filenames in Windows)
- The `runNum` variable now determines how many times the script is run. Runs are done completely seperately, only the hyperparameters remain unchanged (data splitting, normalization and model fitting could all be the cause of other results). The whole script has been customized to this.
- TensorFlow information output is now suppressed; only warnings and errors should be printed (for clarity)
- Updated docstrings



### 6 Sep 2020
- Added function `write_history`: validation and training losses/accuracies are now tracked over every epoch and written to an xlsx-file.
- Added run time to .mat-file.
- Updated docstrings


### 5 Sep 2020
- First commit of all files.

Changes made, compared to the default files:
- `make_label` and `split_data` functions have been created.
- In the `preprocess` function there has been a change on `imgs_p[i] = resize(imgs[i], (IMG_COLS, IMG_ROWS), preserve_range=True)`;
  `IMG_COLS` and `IMG_ROWS` have been switched around, as this would otherwise cause an error.
- `params.py` has been made, containing all the essential parameters and constants.
- Some small changes.
