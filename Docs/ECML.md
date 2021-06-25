# ECML-PKDD documentation

Since the release of the ECML-PKDD paper [insert link when available](), our pipeline has undergone some minor revisions to improve the user experience, and ensure a more reliable framework for the delivery of data and metadata. The latest model training, validation, testing pipelines can be found on [HumBugDB](https://github.com/HumBug-Mosquito/HumBugDB/), which contains detailed instructions for data access. The validity of our method has not been questioned, instead the additions represent an incremental improvement. A full log of changes is included here for completeness:

* Significant cleaning of the data and metadata simplify the data wrangling and cleaning 
* Added support for PyTorch
* Automatically produce broader range of performance metrics when evaluating results

## Legacy code for reproduction
The model used in the publication is found at:
```
Code/models/BNN/Win_40_Stride_5_CNN_log-mel_128_norm_Falseheld_out_test_manual_v2_low_epoch.h5
```
If you wish to use this with the prediction pipeline in this repository, you must change the window size parameter `--win_size` to 40. The new model used a window size of 1.92 seconds instead of 2.56, as additional data became available often with a label duration which was shorter than our old window size. Performance difference appears minor.

The data processing of old database data, the model training, and evaluation are included in:
```
Code/notebooks/detector_testing_full.ipynb
```
A Jupyter notebook for using the detection pipeline, along with helper functions which extract metadata and collate the predictions from multiple recordings (used extensively for the creation of Table 1)  is given in:
```
Code/notebooks/VAD_detector_pipeline.ipynb
```

 
