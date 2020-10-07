import matplotlib.pyplot as plt
import librosa
import numpy as np


# Dataframe processing and File i/o

def remove_label_bug(df, verbose=0):
    '''Remove any entries where the label end_time is not at least 1 sample greater than the label start_time.
    This step can be done more efficiently when loading data into .wav files, but for readability and ease of use we will
    pre-process the dataframes first by removing any entries with incorrect labels.'''
    idx_noise = []
    for i, item in enumerate(range(len(df))):
        # Check for entry errors
        label_duration = df.iloc[i]["fine_end_time"] - df.iloc[i]["fine_start_time"]
        if label_duration <= 0:
            if verbose==2:
                print('Label duration of', label_duration, 'seconds at path', df.iloc[i]["path"], '... deleting index', i)
            idx_noise.append(i)
    df_clean = df.drop(index = df.index[idx_noise])
    if verbose>=1:
        print('Old dataframe length:', len(df))
        print('New dataframe length:', len(df_clean))
    return df_clean


def get_wav_for_df(df, sr):
    '''Extract wave files from dataframe df, at signal rate sr. Returns a list of numpy arrays in float 32, and the signal length'''
    # Probably memory inefficient as it loops over entire paths and loads all the corresponding wav file excerpts, may be better to open file
    # and perform feature transform, close, saving as we progress
    x = []
    idx = []
    signal_length = 0
    # Load data at the resampled rate of sr Hz
    for i, item in enumerate(range(len(df))):

        signal, _ = librosa.load('../../Data' + df.iloc[i]["path"].strip().strip("\'"),
                    offset=df.iloc[i]["fine_start_time"],
                    duration=df.iloc[i]["fine_end_time"] - df.iloc[i]["fine_start_time"], sr = sr)
        x.append(signal)
        signal_length += df.iloc[i]["fine_end_time"] - df.iloc[i]["fine_start_time"]
    return x, signal_length


def get_wav_for_path(noise_path_names, sr):
    x = []
    signal_length = 0
    for path in noise_path_names:
        signal, _ = librosa.load('../../Data' + path, sr=sr)
        x.append(signal)
        signal_length += len(signal)/sr
    return x, signal_length


def get_noise_wav_for_df(df, path_strings, crop_time, sr, verbose=0):
    ''' Logic: If any wav file has a positive label, the unlabelled wav file is negative. 
    This only holds true for some data in the dataset, hence we supply a list of strings to path_string, which filters the 
    data as follows: if path_string is a substring of path, accept this data processing scheme.
    crop time supplied allows to crop the noise window on both ends, to account for the possibility of
    mosquito to be present before the indicated start and after the indicated end time.'''
    x = []
    signal_length = 0
    # Added to let input conform to list type (single or multiple input arguments handled the same way in code)
    path_strings = path_strings if isinstance(path_strings, list) else [path_strings]  
    print(path_strings)
    
    # Load data at the resampled rate of 8000 Hz, which is used in the smartphone app
    for path in df["path"].unique():
        for path_string in path_strings:
#             print(path_string)
            if path_string in path:
                if verbose == 2:
                    print(path)
                df_wav = df[df["path"] == path]
                if verbose == 2:
                    print(df_wav)
                wav = librosa.load('../../Data' + df_wav.iloc[0]["path"].strip().strip("\'"), sr =8000)
                
                idx_wav_array = np.zeros(len(wav[0]), dtype=int)
                sample_start_time = round((df_wav["fine_start_time"] - crop_time) * sr)
                sample_end_time = round((df_wav["fine_end_time"] + crop_time)* sr)
#                 print(sample_end_time-sample_start_time)
                
                if sample_end_time.iloc[len(df_wav)-1] > len(wav[0]):
                    if verbose >= 1:
                        print('Sample end time', sample_end_time.iloc[len(df_wav)-1], 'greater than length',
                              len(wav[0]), 'of the wave file, rounding down label to end of array')
                    sample_end_time.iloc[len(df_wav)-1] = len(wav[0])
                for i in range(len(df_wav)):
                    if verbose == 2:
                        print(sample_start_time.iloc[i], sample_end_time.iloc[i])
                    idx_wav_array[np.arange(sample_start_time.iloc[i], sample_end_time.iloc[i]).astype('int')] = 1
                x.append(wav[0][np.where(np.logical_not(idx_wav_array))])
                signal_length  +=len(wav[0][np.where(np.logical_not(idx_wav_array))])/sr


    return x, signal_length


# Feature processing with Librosa

def get_feat(x, sr, flatten=True):
    ''' Returns features extracted with Librosa. Currently written to support MFCCs only. flatten=True
    returns a continguous list of all the features from different recordings concatenated, whereas flatten=False returns
    a list of features, with the number of items equal to the number of input recordings'''
    X = []
    n_samples = 0
    for audio in x:
        if len(audio) > 0:
            feat = librosa.feature.mfcc(y=np.array(audio), sr=sr, n_mfcc=13)
            X.append(feat)
            n_samples += np.shape(feat)[1]

    if flatten:
        X_flatten = np.zeros([len(X[0]), n_samples])
        curr_idx = 0
        for signal in X:
    #         print(curr_idx, np.shape(signal)[1])
            X_flatten[:,curr_idx:curr_idx+np.shape(signal)[1]] = signal
            curr_idx += np.shape(signal)[1]
        return X_flatten.T
    else:
        return X


# Further evaluation and visualisation

def df_metadata(df, plot=True, filepath=None):
    '''Insert dataframe to calculate quantity of data per species. Expects pandas dataframe df, returns
    species_list, and durations, optionally plots'''

    durations = []
    species_list = []
    for species in df["species"].unique():
        duration = df[df["species"] == species]["fine_end_time"] - df[df["species"] == species]["fine_start_time"]
        species_list.append(species)
        durations.append(np.sum(duration))
    if plot:
        plt.bar(species_list, durations)
        plt.xticks(rotation=90)
        
        plt.ylabel('Time (s)')
#         plt.tight_layout()
        if filepath:
            plt.savefig(filepath, bbox_inches='tight')
        plt.show()
    return species_list, durations

