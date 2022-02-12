import matplotlib.pyplot as plt
import librosa
import numpy as np
import skimage.util

def detect_timestamps_BNN(preds_prob, G_X, U_X, hop_length=512, out_method='per_window', sr=8000):
    ''' Outputs the mean probability per section of predicted audio: consider the case when many
    consecutive windows are positive, we only get to see the average output. When then thresholding the output
    by probability, it may be better to have individual windows so that we can filter out more likely events. However,
    if we want to keep longer complete sections of audio, this may be sufficient (given the already large window size of 
    2.56s'''


    print(np.shape(preds_prob))
    frames = librosa.frames_to_samples(np.arange(len(preds_prob)), hop_length=512) 
    preds_list = []

    for class_idx in range(8):


        preds = np.zeros(len(preds_prob))
        if out_method == 'single':
            for i, pred in enumerate(preds_prob):
                if pred[class_idx] > -1e-6:
                    preds[i] = 1
        elif out_method == 'per_window':
            for i, pred in enumerate(preds_prob):
                preds[i] = preds_prob[i,class_idx]       
        else:
            raise ValueError('Output method not recognised.')
            
                 


         
        sample_start = 0
        prob_start_idx = 0
        

        # mozz_pred_array = []
        for index, frame in enumerate(frames[:-1]):
            if preds[index] != preds[index+1]:
                sample_end = frames[index+1]
                prob_end_idx = index+1
                # print('sample_start', sample_start, prob_start_idx, 
                #  'sample_end', sample_end, prob_end_idx, 'label', preds[index])
                if preds[index] == 1 or out_method == 'per_window':
                    preds_list.append([str(sample_start/sr), str(sample_end/sr),
                                       "{:.4f}".format(np.mean(preds_prob[prob_start_idx:prob_end_idx][:,class_idx])) +
                                      " PE: " + "{:.4f}".format(np.mean(G_X[prob_start_idx:prob_end_idx])) + 
                                      " MI: " + "{:.4f}".format(np.mean(U_X[prob_start_idx:prob_end_idx])) + " class " + str(class_idx)])
                sample_start = frames[index+1]  
                prob_start_idx = index+1     

            elif index+1 == len(frames[:-1]):
                sample_end = frames[index+1]
                prob_end_idx = index+1 
                # print('sample_start', sample_start, 'sample_end', sample_end, 'label', preds[index])
                if preds[index] == 1 or out_method == 'per_window':
                    preds_list.append([str(sample_start/sr), str(sample_end/sr),
                                       "{:.4f}".format(np.mean(preds_prob[prob_start_idx:prob_end_idx][:,class_idx])) +
                                      " PE: " + "{:.4f}".format(np.mean(G_X[prob_start_idx:prob_end_idx])) + 
                                      " MI: " + "{:.4f}".format(np.mean(U_X[prob_start_idx:prob_end_idx])) + " class " + str(class_idx)])       
                sample_start = frames[index+1]       
                prob_start_idx = index+1 
    return preds_list

