import os
import util
import util_dashboard as util_dash
import util_species
import librosa
import numpy as np
import sys
from tensorflow import keras
import argparse
import matplotlib.pyplot as plt

def write_output(rootFolderPath, audio_format,  dir_out=None, det_threshold=0.5, n_samples=10, feat_type='log-mel',n_feat=128, win_size=30, step_size=30,
                 n_hop=512, sr=8000, norm_per_sample=True, debug=False, to_dash=False, out_method='per_window'):

        '''dir_out = None if we want to save files in the same folder that we read from.
           det_threshold=0.5 determines the threshold above which an event is classified as positive. See detect_timestamps for 
           a more nuanced discussion on thresholding and what we wish to save upon running the algorithm.'''
        # rootFolderPath = 'F:\PostdocData\HumBugServer\SemiFieldDataTanzania'
        # audio_format = '.wav'



        model = keras.models.load_model('../models/BNN/Win_30_Stride_5_2021_07_31_03_25_35-e69accuracy0.9608.hdf5', custom_objects={"dropout": 0.2}) # change this here.
        model_name = 'Win_30_Stride_5_2021_07_31_03_25_35-e69accuracy0.9608.hdf5' # change this
        mozz_audio_list = []
        
        print('Processing:', rootFolderPath, 'for audio format:', audio_format)
        i_signal = 0
        for root, dirs, files in os.walk(rootFolderPath):
            for filename in files:	
                if audio_format in filename:
                    print(root, filename) 
                    i_signal+=1
                    try:            
                        x, x_l = util.get_wav_for_path_pipeline([os.path.join(root, filename)], sr =8000)
                        if debug:
                            print(filename + ' signal length', x_l)
                        if x_l < (n_hop * win_size)/sr: 
                            print('Signal length too short, skipping:', x_l, filename) 
                        else:
            #             
                            X_CNN = util.get_feat(x, sr=8000, feat_type=feat_type, n_feat=n_feat,norm_per_sample=norm_per_sample, flatten = False)

                            X_CNN = util.reshape_feat(X_CNN, win_size=win_size, step_size=step_size)
                            out = []
                            for i in range(n_samples):
                                out.append(model.predict(X_CNN))

                            n_classes = np.shape(out)[-1]       
                            G_X, U_X, _ = util.active_BALD(np.log(out), X_CNN, n_classes)


                            y_to_timestamp = np.repeat(np.mean(out, axis=0), step_size, axis=0)
                            print(y_to_timestamp)
                            G_X_to_timestamp = np.repeat(G_X, step_size, axis=0)
                            U_X_to_timestamp = np.repeat(U_X, step_size, axis=0)
                            preds_list = util_species.detect_timestamps_BNN(y_to_timestamp, G_X_to_timestamp, U_X_to_timestamp, 
                                                                out_method=out_method)   

                            if debug:
                                print(preds_list)
                                for times in preds_list:
                                    mozz_audio_list.append(librosa.load(os.path.join(root, filename), offset=float(times[0]),
                                                                     duration=float(times[1])-float(times[0]), sr=sr)[0])


                            if dir_out:
                                root_out = root.replace(rootFolderPath, dir_out)
                            else:
                                root_out = root
                            print('dir_out', root_out, 'filename', filename)
  

                            if not os.path.exists(root_out): os.makedirs(root_out)

                            if filename.endswith(audio_format):  
                                output_filename = filename[:-4]  # remove file extension for renaming to other formats.
                            else:
                                output_filename = filename # no file extension present

                            text_output_filename = os.path.join(root_out, output_filename) + '_BNN_step_' + str(step_size) + '_samples_' + str(n_samples) + '_'+ str(model_name) + '.txt'
                            np.savetxt(text_output_filename, preds_list, fmt='%s', delimiter='\t')

                            if not to_dash: 
                                mozz_audio_filename, audio_length, has_mosquito = util_dash.write_audio_for_plot(text_output_filename, root, filename, output_filename, root_out, sr)
                                if has_mosquito:
                                    plot_filename = util_dash.plot_mozz_MI(X_CNN, y_to_timestamp[:,1], U_X_to_timestamp, 0.5, root_out, output_filename)
                                    util_dash.write_video_for_dash(plot_filename, mozz_audio_filename, audio_length, root_out, output_filename)
                    except Exception as e:
                        print("[ERROR] Unable to load {}".format(os.path.join(root, filename)))
                        print(e)

        print('Total files of ' + str(audio_format) + ' format processed:', i_signal)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This function writes the predictions of the model.
    """)
    parser.add_argument("rootFolderPath", help="Source destination of audio files. Can be a parent directory.")
    parser.add_argument("audio_format", help="Any file format supported by librosa load.")
    parser.add_argument("--dir_out", help="Output directory. If not specified, predictions are output to the same folder as source.")
    parser.add_argument("--no_vid", action='store_true', help="Save predicted audio, video, and corresponding labels to same directory as dictated by dir_out.")
    parser.add_argument("--norm", default=True, help="Normalise feature windows with respect to themsleves.")
    parser.add_argument("--win_size", default=30, type=int, help="Window size.")
    parser.add_argument("--step_size", default=30, type=int, help="Step size.")
    parser.add_argument("--BNN_samples", default=10, type=int, help="Number of MC dropout samples.")
    parser.add_argument("--out_method", default='per_window', type=str, help='''Output format of labels. `per_window` outputs probability of each 
        class per window. `single` outputs a single prediction per audio signal.''')


    # dir_out=None, det_threshold=0.5, n_samples=10, feat_type='log-mel',n_feat=128, win_size=40, step_size=40,
    #              n_hop=512, sr=8000, norm=False, debug=False, to_filter=False

    args = parser.parse_args()

    rootFolderPath = args.rootFolderPath
    audio_format = args.audio_format
    dir_out = args.dir_out
    no_vid = args.no_vid
    win_size = args.win_size
    step_size = args.step_size
    n_samples = args.BNN_samples
    norm_per_sample=args.norm
    out_method=args.out_method


    write_output(rootFolderPath, audio_format, dir_out=dir_out, norm_per_sample=norm_per_sample,
                 win_size=win_size, step_size=step_size, to_dash=no_vid, n_samples=n_samples,out_method=out_method)