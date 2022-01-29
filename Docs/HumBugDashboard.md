# Predictions for HumBug Pipeline
## Instructions for regular maintenance and running of mosquito predictions for dashboard

See the main readme [link] for a description of general use and function arguments. This documentation is intended to describe how to regularly obtain predictions for use with the HumBug dashboard.

1. `mlrg-humbug-gpu` is the machine (current as of 29/01/2022) which houses this repository. The database is mounted from `rvm7`. All predictions are carried out on filepaths relative to the mounting location.
2. The commands are executed from `/home/*user_name*/DatabaseAccess/Code/lib`, in the `conda` environment `BNN-gpu-py37`.
3. Add `export PATH=/opt/anaconda/anaconda3/bin:$PATH` to `/home/*user_name*/.bashrc` and `source ~/.bashrc` to load the environment in the current shell
4. A prediction for all audio in a single day can be made with the following example command:
``` example command ```
5. You may also use a script to automate command execution for several root folders, as can be found in `lib/pred_script.sh`. For example, to predict for all data with the datestamp of the 1st to 30th of November 2030, run:

```#!/bin/bash
for i in {01..30}
do
        python -W ignore predict.py --dir_out /home/ivank/dbmount/MozzWearPlot/2030-11-$i/ --to_dash True /home/ivank/dbmount/MozzWear/2030-11-$i/ .aac
done 
```
