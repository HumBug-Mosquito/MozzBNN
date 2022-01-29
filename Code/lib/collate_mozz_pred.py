import pandas as pd
import os 
import argparse

def find_mozz_pred(df_path, file_out):
    df = pd.read_csv(df_path)
    df['mozz_pred_wav'] = False
    df['mozz_pred_labels'] = False
    file_count = 0

    for idx, row in df.iterrows():
        path = row['path'].replace('/data/MozzWear', '/home/ivank/dbmount/MozzWearPlot')
        # if wave file of positive predictions exists, append the wave file and labels to dataframe:
        if os.path.isfile(path + '_mozz_pred.wav'):
            file_count+=1
            df.loc[idx, 'mozz_pred_wav'] = path + '_mozz_pred.wav'
            df.loc[idx, 'mozz_pred_labels'] = path + '_mozz_pred.txt'
    print('Found files with mosquito predictions:', file_count)
    df.to_csv(file_out, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Utility to help find predictions from original waveform files in the form of `path_filename.wav`.
    Currently hardcoded to edit paths from locations of data on server, but may be modified to any.""")
    parser.add_argument("dataframeSource", help="Source dataframe containing column `path` with filenames of candidate recordings.")
    parser.add_argument("csvOut", help="Filename, including path, to save the original dataframe with an extra columns corresponding to found labels and predictions")
   
    args = parser.parse_args() 
    dataframeSource = args.dataframeSource
    csvOut = args.csvOut
    find_mozz_pred(dataframeSource,csvOut)
