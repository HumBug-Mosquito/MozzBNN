import matplotlib.pyplot as plt
import librosa
import numpy as np
import skimage.util


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

def get_wav_for_path_pipeline(path_names, sr):
    x = []
    signal_length = 0
    for path in path_names:
#         print(path)
        signal, _ = librosa.load(path, sr=sr)
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

def get_feat(x, sr, feat_type, n_feat, flatten=True):
    ''' Returns features extracted with Librosa. Currently written to support MFCCs (truncated), MFCCs, and log-mel only. flatten=True
    returns a continguous list of all the features from different recordings concatenated, whereas flatten=False returns
    a list of features, with the number of items equal to the number of input recordings'''
    X = []
    n_samples = 0
    for audio in x:
        if len(audio) > 0:
            if feat_type == 'mfcc':
                feat = librosa.feature.mfcc(y=np.array(audio), sr=sr, n_mfcc=n_feat)
            elif feat_type == 'mfcc-cut':
                feat = librosa.feature.mfcc(y=np.array(audio), sr=sr, n_mfcc=n_feat)[2:]
            elif feat_type == 'log-mel':
                feat = librosa.feature.melspectrogram(y=np.array(audio), sr=sr, n_mels=n_feat)
                # Added case to present features in decibels:
                feat = librosa.power_to_db(feat, ref=np.max)
            else:
                raise Exception('Feature type of log-mel, mfcc-cut, or mfcc only supported.')

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


def reshape_feat(feats, win_size, step_size):
    '''Reshaping features from get_feat to be compatible for classifiers expecting a 2D slice as input. Parameter `win_size` is 
    given in number of feature windows. Can code to be a function of time and hop length instead in future.'''
    
    feats_windowed_array = []
    for feat in feats:
        if np.shape(feat)[1] < win_size:
            pass
        else:
            feats_windowed = skimage.util.view_as_windows(feat.T, (win_size,np.shape(feat)[0]), step=step_size)
            feats_windowed_array.append(feats_windowed)
    return np.vstack(feats_windowed_array)





# Return predicted sections in time from features:


def detect_timestamps(preds_prob, hop_length=512, det_threshold = 0.5, sr=8000):

    preds = np.zeros(len(preds_prob))
    for i, pred in enumerate(preds_prob):
        if pred[1] > det_threshold:
            preds[i] = 1


    frames = librosa.frames_to_samples(np.arange(len(preds)), hop_length=512)  
    sample_start = 0
    prob_start_idx = 0
    preds_list = []
    # mozz_pred_array = []
    for index, frame in enumerate(frames[:-1]):
        if preds[index] != preds[index+1]:
            sample_end = frames[index+1]
            prob_end_idx = index+1
            # print('sample_start', sample_start, prob_start_idx, 
            #  'sample_end', sample_end, prob_end_idx, 'label', preds[index])
            if preds[index] == 1:
                preds_list.append([sample_start/sr, sample_end/sr, "{:.2f}".format(np.mean(preds_prob[prob_start_idx:prob_end_idx][:,1]))])
            sample_start = frames[index+1]  
            prob_start_idx = index+1     

        elif index+1 == len(frames[:-1]):
            sample_end = frames[index+1]
            prob_end_idx = index+1 
            # print('sample_start', sample_start, 'sample_end', sample_end, 'label', preds[index])
            if preds[index] == 1:
                preds_list.append([sample_start/sr, sample_end/sr, "{:.2f}".format(np.mean(preds_prob[prob_start_idx:prob_end_idx][:,1]))])
            sample_start = frames[index+1]       
            prob_start_idx = index+1 
    return preds_list



def detect_timestamps_BNN(preds_prob, G_X, U_X, hop_length=512, det_threshold=0.5, sr=8000):
    ''' Outputs the mean probability per section of predicted audio: consider the case when many
    consecutive windows are positive, we only get to see the average output. When then thresholding the output
    by probability, it may be better to have individual windows so that we can filter out more likely events. However,
    if we want to keep longer complete sections of audio, this may be sufficient (given the already large window size of 
    2.56s'''
    preds = np.zeros(len(preds_prob))
    for i, pred in enumerate(preds_prob):
        if pred[1] > det_threshold:
            preds[i] = 1


    frames = librosa.frames_to_samples(np.arange(len(preds)), hop_length=512)  
    sample_start = 0
    prob_start_idx = 0
    preds_list = []
    # mozz_pred_array = []
    for index, frame in enumerate(frames[:-1]):
        if preds[index] != preds[index+1]:
            sample_end = frames[index+1]
            prob_end_idx = index+1
            # print('sample_start', sample_start, prob_start_idx, 
            #  'sample_end', sample_end, prob_end_idx, 'label', preds[index])
            if preds[index] == 1:
                preds_list.append([str(sample_start/sr), str(sample_end/sr),
                                   "{:.4f}".format(np.mean(preds_prob[prob_start_idx:prob_end_idx][:,1])) +
                                  " PE: " + "{:.4f}".format(np.mean(G_X[prob_start_idx:prob_end_idx])) + 
                                  " MI: " + "{:.4f}".format(np.mean(U_X[prob_start_idx:prob_end_idx]))])
            sample_start = frames[index+1]  
            prob_start_idx = index+1     

        elif index+1 == len(frames[:-1]):
            sample_end = frames[index+1]
            prob_end_idx = index+1 
            # print('sample_start', sample_start, 'sample_end', sample_end, 'label', preds[index])
            if preds[index] == 1:
                preds_list.append([str(sample_start/sr), str(sample_end/sr),
                                   "{:.4f}".format(np.mean(preds_prob[prob_start_idx:prob_end_idx][:,1])) +
                                  " PE: " + "{:.4f}".format(np.mean(G_X[prob_start_idx:prob_end_idx])) + 
                                  " MI: " + "{:.4f}".format(np.mean(U_X[prob_start_idx:prob_end_idx]))])       
            sample_start = frames[index+1]       
            prob_start_idx = index+1 
    return preds_list










# Bayesian Neural Network

def active_BALD(out, X, n_classes):

    log_prob = np.zeros((out.shape[0], X.shape[0], n_classes))
    score_All = np.zeros((X.shape[0], n_classes))
    All_Entropy = np.zeros((X.shape[0],))
    for d in range(out.shape[0]):
#         print ('Dropout Iteration', d)
#         params = unflatten(np.squeeze(out[d]),layer_sizes,nn_weight_index)
        log_prob[d] = out[d]
        soft_score = np.exp(log_prob[d])
        score_All = score_All + soft_score
        #computing F_X
        soft_score_log = np.log2(soft_score+10e-15)
        Entropy_Compute = - np.multiply(soft_score, soft_score_log)
        Entropy_Per_samp = np.sum(Entropy_Compute, axis=1)
        All_Entropy = All_Entropy + Entropy_Per_samp
 
    Avg_Pi = np.divide(score_All, out.shape[0])
    Log_Avg_Pi = np.log2(Avg_Pi+10e-15)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
    G_X = Entropy_Average_Pi
    Average_Entropy = np.divide(All_Entropy, out.shape[0])
    F_X = Average_Entropy
    U_X = G_X - F_X
# G_X = predictive entropy
# U_X = MI
    return G_X, U_X, log_prob




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

