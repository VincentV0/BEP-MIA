# Changelog

### 6 Sep 2020
- Added function 'write_history': validation and training losses/accuracies are now tracked over every epoch and written to an xlsx-file.
- Added run time to .mat-file.
- Updated docstrings


### 5 Sep 2020
- First commit of all files.

Changes made, compared to the default files:
- 'make_label' and 'split_data' functions have been created.
- In the 'preprocess' function there has been a change on "imgs_p[i] = resize(imgs[i], (IMG_COLS, IMG_ROWS), preserve_range=True)";
  IMG_COLS and IMG_ROWS have been switched around, as this would otherwise cause an error.
- "params.py" has been made, containing all the essential parameters and constants.
- Some small changes.
