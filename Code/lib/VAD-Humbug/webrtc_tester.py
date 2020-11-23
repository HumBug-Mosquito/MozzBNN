# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 21:29:33 2020

Script for showing the effectiveness of the WebRTC VAD speech-removal pipeline
with a variety of files

@author: benpg
"""

from webrtc_vad import VAD
from webrtc_utils import plt_speech, speech_stats
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import librosa
from VAD_pipeline import VAD_pipeline


def webrtc_tester(audiodir, labelspath, aggressiveness, norm_LU=False, plots_on=False):
    '''
    Parameters
    ----------
    audiodir : str
        Path to folder containing audio files to analyse.
    labelspath : str
        Path to folder containing speech label txt files with identical names to the .wav files.
    aggressiveness : int
        Aggressiveness with which noise is removed from file (0-3).
    norm_LU : int or False, optional
        Takes a value that must be a negative integer as the target LU at which to loudness normalise the file (dB). The default is False, i.e. no normalisation.
    plots_on : bool or int, optional
        Decides if/how many plots of true/predicted speech to plot for the files given. False: no plots (default); True: plot all files; integer n: plot the first n files
        
    Returns
    -------
    true_speech: list of floats
        List with each element the number of seconds of correctly identified speech in each file.
    speech_total_length: list of floats
        List with each element the total number of seconds of speech in each file, given by the speech labels
    true_noise: list of floats
        List with each element the number of seconds of correctly identified noise in each file.
    noise_total_length: list of floats
        List with each element the total number of seconds of noise in each file.

    '''
    pltcount = 0
    true_speech, speech_total_length, true_noise, noise_total_length = [],[],[],[]
    if plots_on==True: plots_on = np.inf     # plot all files if no limit set
    if norm_LU:
        normdir = join(audiodir,'norm')
        if not os.path.exists(normdir): os.mkdir(normdir)
        for file in os.listdir(audiodir):
            if file.endswith('.wav'):
                VAD_pipeline(audiodir, file, normdir, agg=aggressiveness, sr=8000, LU=norm_LU, norm_only=True) 
        audiodir = normdir

    for file in os.listdir(audiodir):
        if file.endswith('.wav'):
            audio, sr = librosa.load(join(audiodir,file), sr=2000)
            speech_pred_labels = VAD(join(audiodir,file), aggressiveness, chunks=False)
            filename = file.rstrip('.wav')
            try:
                speech_labels = np.loadtxt(join(labelspath,filename)+'.txt', usecols=(0,1))
            except:
                # If label file not found, print warning and process with no speech labels
                print('\nFile: %s\nLabel file not found, assuming no speech.\n' % file)
                speech_labels = np.array([[0,0]])
            file_length = audio.shape[0]/sr
            
            # reshape to 2D for files with only one pair of labels
            if speech_labels.shape == (2,): speech_labels = np.array([speech_labels])
            
            # Plot the true and predicted speech
            if pltcount < plots_on:
                plt_speech(speech_pred_labels, speech_labels, file_length, filename, audio, sr)
                pltcount += 1
            
            # Get and print test information
            print('\nFile:', file)
            stats = speech_stats(speech_pred_labels, speech_labels, file_length)
            # Keep running total of length of speech and length predicted correctly
            true_speech.append(stats[0])
            speech_total_length.append(stats[1])
            true_noise.append(stats[2])
            noise_total_length.append(stats[3])
    return(true_speech, speech_total_length, true_noise, noise_total_length)


plt.close('all')

aggressiveness = 0
plots_on = True  # Plot true/predicted speech. False for none, True for all, or an int number
norm_LU = -35     # False or a number

path = r'D:\Big Files\Humbug OneDrive\Experiments\IHI_small_sample'
labelspath = join(path, 'speech_labels')

audiodir = path

true_speech, speech_total_length, true_noise, noise_total_length = \
    webrtc_tester(audiodir, labelspath, aggressiveness, norm_LU, plots_on)

tpr_overall = np.sum(true_speech)/np.sum(speech_total_length)
tnr_overall = np.sum(true_noise)/np.sum(noise_total_length)
print('\nOVERALL:\n%.4f percent of speech detected (%.2f out of %.2f seconds)' \
      % (100*tpr_overall, np.sum(true_speech), np.sum(speech_total_length)))
print('%.4f percent of noise preserved (%.2f out of %.2f seconds)' \
  % (100*tnr_overall, np.sum(true_noise), np.sum(noise_total_length)))