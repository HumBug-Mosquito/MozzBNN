import os
import util
import librosa
import numpy as np
import sys
from tensorflow import keras
import argparse
import matplotlib.pyplot as plt

def write_output(rootFolderPath, audio_format,  dir_out=None, det_threshold=0.5, n_samples=10, feat_type='log-mel',n_feat=128, win_size=40, step_size=40,
                 n_hop=512, sr=8000, norm=False, debug=False, to_filter=False, plot=None):

        '''dir_out = None if we want to save files in the same folder that we read from.
           det_threshold=0.5 determines the threshold above which an event is classified as positive. See detect_timestamps for 
           a more nuanced discussion on thresholding and what we wish to save upon running the algorithm.'''
        # rootFolderPath = 'F:\PostdocData\HumBugServer\SemiFieldDataTanzania'
        # audio_format = '.wav'





        model = keras.models.load_model('../models/BNN/Win_40_Stride_5_CNN_log-mel_128_norm_Falseheld_out_test_manual_v2_low_epoch.h5',
                                       custom_objects={"dropout": 0.2})
        model_name = 'held_out_test_manual_v2_low_epoch'

        mozz_audio_list = []

        i_signal = 0
        for root, dirs, files in os.walk(rootFolderPath):
            for filename in files:

                if audio_format in filename:
                    print(root, filename) 
                    i_signal+=1
            
                    x, x_l = util.get_wav_for_path_pipeline([os.path.join(root, filename)], sr =8000)
                    if debug:
                        print(filename + ' signal length', x_l)
                    if x_l < (n_hop * win_size)/sr: 
                        print('Signal length too short, skipping:', x_l, filename) 
                    else:
            #             
                        X_CNN = util.get_feat(x, sr=8000, feat_type=feat_type, n_feat=n_feat, flatten = False)

                        X_CNN = util.reshape_feat(X_CNN, win_size=win_size, step_size=step_size)
            #             X_CNN = (X_CNN - mean)/std
            #             print(np.shape(X_CNN))

                        out = []
                        for i in range(n_samples):
                            out.append(model.predict(X_CNN))

                        G_X, U_X, _ = util.active_BALD(np.log(out), X_CNN, 2)
                        preds_list = util.detect_timestamps_BNN(np.repeat(np.mean(out, axis=0), step_size, axis=0),
                                              np.repeat(G_X, step_size, axis=0),
                                              np.repeat(U_X, step_size, axis=0), det_threshold=det_threshold)   

                        if to_filter:
                            preds_filt = np.zeros([len(preds_CNN),2])
                            preds_filt[:,1] = medfilt(preds_CNN[:,1], kernel_size=51)
                            preds_filt[:,0] = 1 - preds_filt[:,1]
                            preds_CNN = preds_filt

                        if debug:
                            print(preds_list)
                            for times in preds_list:
                                mozz_audio_list.append(librosa.load(os.path.join(root, filename), offset=float(times[0]),
                                                                     duration=float(times[1])-float(times[0]), sr=sr)[0])
        #                     plt.plot((1/sr)*librosa.frames_to_samples(np.arange(len(np.repeat(np.mean(out, axis=0),step_size, axis=0))), hop_length=512),
        #                                                    np.repeat(np.mean(out, axis=0), step_size, axis=0)[:,1])
        #                     plt.xlabel('Seconds')
        #                     plt.ylabel('Mean probability (mozz)')
        #                     plt.show()

                        if dir_out:
                            root_out = root.replace(rootFolderPath, dir_out)
                        print('dir_out', root_out, 'filename', filename)
  

                        if not os.path.exists(root_out): os.makedirs(root_out)
                        np.savetxt(os.path.join(root_out, filename) + '_BNN_step_' + str(step_size) + '_samples_' + str(n_samples) + '_'
                                + str(model_name) + '.txt', preds_list, fmt='%s', delimiter='\t')

                        if plot:
                            plt.figure(figsize=(10,5))
                            plt.plot((1/sr)*librosa.frames_to_samples(np.arange(len(np.repeat(np.mean(out, axis=0),step_size, axis=0))), hop_length=n_hop),
                                                            np.repeat(np.mean(out, axis=0), step_size, axis=0)[:,1])
                            plt.xlabel('Seconds')
                            plt.ylabel('Mean probability (mozz)')
                            plt.tight_layout()
                            plt.savefig(os.path.join(root_out, filename) + '.svg')
                            plt.close(plt.gcf()) # May be better to re-write to not use plt API


        print('Total files of ' + str(audio_format) + ' format processed:', i_signal)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This function writes the predictions of the model.
    """)
    parser.add_argument("rootFolderPath", help="Source destination of audio files. Can be a parent directory.")
    parser.add_argument("audio_format", help="Any file format supported by librosa load.")
    parser.add_argument("--dir_out", help="Output directory. If not specified, predictions are output to the same folder as source.")
    parser.add_argument("--plot", help="Save figure of predictions to same directory as dictated by dir_out.")
    parser.add_argument("--win_size", default=40, type=int, help="Window size.")
    parser.add_argument("--step_size", default=40, type=int, help="Step size.")


    # dir_out=None, det_threshold=0.5, n_samples=10, feat_type='log-mel',n_feat=128, win_size=40, step_size=40,
    #              n_hop=512, sr=8000, norm=False, debug=False, to_filter=False

    args = parser.parse_args()

    rootFolderPath = args.rootFolderPath
    audio_format = args.audio_format
    dir_out = args.dir_out
    plot = args.plot
    win_size = args.win_size
    step_size = args.step_size


    write_output(rootFolderPath, audio_format, dir_out=dir_out, win_size=win_size, step_size=step_size, plot=plot)
