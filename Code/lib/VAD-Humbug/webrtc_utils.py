# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 08:05:07 2020

@author: benpg
"""
#webrtc utils

import numpy as np
import matplotlib.pyplot as plt

def plt_speech(speech_pred_labels, speech_labels, file_length, filename, audio, sr):
    '''
    Parameters
    ----------
    speech_pred_labels : n*2 numpy array
        Start and end timestamps in seconds predicted by WebRTC VAD.
    speech_labels : n*2 numpy array
        True start and end timestamps, loaded from label .txt file.
    file_length : float
        Total length of file in seconds.
    filename : string
        Name of file.
    audio : 1D numpy array
        Time series audio sampled at sr.
    sr : int
        Sampling frequency.

    Returns
    -------
    None. Plots true and predicted speech, and the raw audio itself, sampled at 2KHz to reduce to a manageable number of points.
    ''' 
    
    plt.figure(filename)
    
    # plot true sections
    for i in range(speech_labels.shape[0]):
        plt.axvspan(speech_labels[i,0], speech_labels[i,1], 0, 1, color='c', alpha=0.5)
    # plot predicted sections
    for i in range(speech_pred_labels.shape[0]):
        plt.axvspan(speech_pred_labels[i,0], speech_pred_labels[i,1], 0, 1, color='y', alpha=0.5)
    # ensure entire length of file is displayed
    plt.axvspan(file_length, file_length, 0,1)    
    # plot audio signal
    x = audio.shape[0]/sr
    plt.plot(np.linspace(0,x,audio.shape[0]), audio, color='r', alpha=1)
    # labels and legends
    plt.xlabel('seconds')
    plt.title(filename)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='c', lw=4),
                    Line2D([0], [0], color='y', lw=4)]
    # plt.title(transcript_path)
    plt.legend(custom_lines, ['true speech labels', 'predicted speech labels'])
    

def speech_stats(speech_pred_labels, speech_labels, file_length):
    '''

    Parameters
    ----------
    speech_pred_labels : n*2 numpy array
        Start and end timestamps in seconds predicted by WebRTC VAD.
    speech_labels : n*2 numpy array
        True start and end timestamps, loaded from label .txt file.
    file_length : float
        Total length of file in seconds.
    
    Returns
    -------
    true_speech: float
        Number of seconds of correctly identified speech.
    speech_total_length:
        Total number of seconds of speech in file given by .txt label file.
    
    Prints results of test, comparing predicted with true, and a confusion matrix.

    '''

    true_speech, speech_pred_total_length = 0, 0
    incident_count = np.zeros((speech_labels.shape[0],))
    speech_total_length = np.sum(speech_labels[:,1] - speech_labels[:,0])
    for i in range(speech_pred_labels.shape[0]):    # all predicted incidents
        lb = speech_pred_labels[i,0]
        ub = speech_pred_labels[i,1]
        speech_pred_total_length = speech_pred_total_length + ub - lb
        for j in range(speech_labels.shape[0]):     # all true incidents
            start = speech_labels[j,0]
            end = speech_labels[j,1] 
            # checking for overlap
            if (start<=lb and end>=ub) or (lb<=start<=ub or lb<=end<=ub):
                # counts correctly identified seconds
                true_speech = true_speech + np.min((end, ub)) - np.max((start,lb))
                incident_count[j] = 1    # counts a success if any overlap
        
    # Confusion Matrix - defining 9 values in the grid
    q1 = true_speech
    q3 = speech_pred_total_length
    q2 = q3 - q1
    q7 = speech_total_length
    q9 = file_length
    q6 = q9 - q3
    q4 = q7 - q1
    q5 = q6 - q4
    q8 = q9 - q7
    
    # printing confusion matrix    
    print('%12s%12s%12s' % ('', 'True Speech', 'True Noise'))
    print('%12s%12.2f%12.2f | %6.2f' % ('Pred Speech', q1,q2,q3))
    print('%12s%12.2f%12.2f | %6.2f' % ('Pred Noise', q4,q5,q6))
    print('%13s'%'' + '-'*33)
    print('%12s%12.2f%12.2f | %6.2f' % ('', q7,q8,q9))
    
    
    # Print some test statistics
    
    # How much of the file overall is predicted speech compared with groundtruth
    print("Predicted %.f percent of file as speech." % \
          (100*speech_pred_total_length/file_length))
    print("In reality %.f percent of file is speech." % \
          (100*speech_total_length/file_length))
       
    # True positive (speech) rates, in both speech 'incidents' and no. of seconds  
    speech_incident_TPR = np.sum(incident_count)/speech_labels.shape[0]
    print('%.2f percent of speech sounds *incidents* correctly classified (%d out of %d)'\
          % (speech_incident_TPR*100, np.sum(incident_count), speech_labels.shape[0]))
    speech_length_TPR = true_speech/speech_total_length
    print('%.2f percent of speech sound in seconds correctly classified (%.2f seconds out of %.2f)'\
          % (speech_length_TPR*100, true_speech, speech_total_length))
    
    true_noise = q5
    noise_total_length = q8
    return(true_speech, speech_total_length, true_noise, noise_total_length)


def speech_stripper(t_series, labels, sr=8000, Lbuffer=0.0, Rbuffer=0.0):
    '''

    Parameters
    ----------
    t_series : 1D numpy array
        Time series input audio.
    labels : n*2 numpy array
        Speech labels in seconds.
    sr : int, optional
       Sampling frequency. The default is 8000.
    Lbuffer : float, optional
        Number of seconds of buffer to add before the cut. The default is 0.0.
    Rbuffer : float, optional
        Number of seconds of buffer to add after the cut. The default is 0.0.

    Returns
    -------
    stripped_t : 1D numpy array
        Output time-series audio with speech removed.
    speech_t : 1D numpy array
        Output time-series audio, speech only.

    '''
    # initialise time series labels and processed time-series audio files
    t_series_labels = np.round(labels * sr)
    stripped_t = np.copy(t_series)
    speech_t = []
    # for each instance of speech
    for i in range(labels.shape[0]):
        # start and end time-series indices for speech, with leeway included
        start = int(np.max((int(t_series_labels[i,0]) - 
                            np.round(sr*Lbuffer), 0)))
        end = int(np.min((t_series_labels[i,1] + 
                          np.round(sr*Rbuffer), t_series.shape[0])))
        length = end - start
        stripped_t[start:end] = np.zeros(length)    # set speech parts to zero
        speech_t.append(t_series[start:end])        # save speech parts
    stripped_t = stripped_t[stripped_t != 0]        # remove speech silent parts
    speech_t = np.concatenate((speech_t))
    return stripped_t, speech_t
        