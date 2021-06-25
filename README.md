# MozzBNN
## A Bayesian Convolutional Neural Network for acoustic mosquito prediction
A BCNN prediction pipeline to discover mosquito sounds from audio.

Model training, validation and testing strategy, and data available on: [HumBugDB](https://github.com/HumBug-Mosquito/HumBugDB).

By Ivan Kiskin. Contact `ivankiskin1@gmail.com` for enquiries or suggestions.
### Installation
```
git clone https://github.com/HumBug-Mosquito/DatabaseAccess.git
```

Requirements:
`condarequirements.txt`, `piprequirements.txt`.

If you experience trouble with the installation of dependent packages, e.g. `numba` decorators for the latest version of `librosa`, you may try:

```
conda install -c anaconda keras-gpu
conda install -c conda-forge librosa=0.7.2 
pip install numba==0.48
```


### How to run
`predict.py` is a command-line utility which accepts the arguments as given in `python predict.py -h`.
From the directory `Code/lib`, execute in the command line `python predict.py [Optional arguments] rootFolderPath audio_format`.

#### Parameters
Required parameters are the source directory `rootFolderPath` which contains audio files of format `audio_format`. Any files of that file format in any subdirectory will be analysed. Outputs are written to the optional argument `--dir_out`. The directory structure of the input will be mirrored with the root input directory replaced by `--dir_out`. If left blank, outputs are written to the same folders which contain audio. For a full list of optional parameters run `python predict.py -h`.

### Model output
By default, the model outputs a text file of mosquito candidates with rows of the form `start_time stop_time   probability predictive_entropy mutual_information`. If you would like to generate an audio file which concatenates all of the detected segments to a new audio file, and parse meta labels to this file, set `--to_dash = True`. The labels were designed for import to [Audacity](https://www.audacityteam.org/) using the label import function in Audacity. The user has three options for visualising predictions:

1. **Audacity**: load the original audio `filename.wav`, and import corresponding label predictions `filename+model_meta_information.txt`
2. **Audacity**: load the detected mosquito candidates under `filename_mozz_pred.wav`, and import corresponding label predictions `filename_mozz_pred.txt`
3. **Any media player**: load the generated mp4 video with mosquito candidates under `filename_mozz_pred.mp4`


### Structure
`Code/lib/` contains `predict.py`, `util.py`, and `util_dashboard.py`
`Code/data/` contains some example audio files to verify the model is working as intended.
`Code/models/` contains a wide range of models that have been trained. By default, the model which performed best on our testbed is used.

### ECML-PKDD 2021
If you are accessing this repo after discovering our ECML publication, [Automatic Acoustic Mosquito Tagging with Bayesian Neural Networks](tbc), and you wish to exactly replicate the experiments for the model in the paper, please see following [documentation](https://github.com/HumBug-Mosquito/MozzBNN/blob/master/Docs/ECML.md). Since acceptance in April 2021, the models and their training framework have been upgraded and improved. The latest model is included here as `Code/models/BNN/neurips_2021_humbugdb_keras_bnn_best.hdf5`, whereas the ECML paper describes `Code/models/BNN/Win_40_Stride_5_CNN_log-mel_128_norm_Falseheld_out_test_manual_v2_low_epoch.h5`. We strongly encourage you to visit [HumBugDB](https://github.com/HumBug-Mosquito/HumBugDB) for the most up-to-date data and training strategy. 



### Additional Documentation:
* [ECML-PKDD model training, validation, testing (deprecated)](https://github.com/HumBug-Mosquito/MozzBNN/blob/master/Docs/ECML.md)
* [Database access (deprecated)](https://github.com/HumBug-Mosquito/DatabaseAccess/blob/master/Docs/legacy_database.md). Please visit latest version on: [HumBugDB](https://github.com/HumBug-Mosquito/HumBugDB).

