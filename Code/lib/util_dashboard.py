import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import librosa

# Change plot properties

COLOR = 'white'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
mpl.rcParams['savefig.facecolor'] = '#2b3e50'

def plot_mozz_MI(X_CNN, y, MI, p_threshold, root_out, filename, out_format='.png'):
    '''Produce plot of all mosquito detected above a p_threshold. Supply Mutual Information values MI, feature inputs 
    X_CNN, and predictions y (1D array of 0/1s). Plot to be displayed on dashboard either via svg or as part of a
    video (looped png) with audio generated for this visual presentation.
    
    `out_format`: .png, or .svg
    
    '''
    pos_pred_idx = np.where(y>p_threshold)[0]

    fig, axs = plt.subplots(2, sharex=True, figsize=(10,5), gridspec_kw={
           'width_ratios': [1],
           'height_ratios': [2,1]})
    # x_lims = mdates.date2num(T)
    # date_format = mdates.DateFormatter('%M:%S')
    # axs[0].xaxis_date()
    # axs[0].xaxis.set_major_formatter(date_format)
    
    axs[0].set_ylabel('Frequency (kHz)')
    
    axs[0].imshow(np.hstack(X_CNN).squeeze().T[:,pos_pred_idx], aspect='auto', origin='lower',
                  extent = [0, len(pos_pred_idx), 0, 4], interpolation=None)
    axs[1].plot(y[pos_pred_idx], label='Probability of mosquito')
    axs[1].plot(MI[pos_pred_idx], '--', label='Uncertainty of prediction')
    axs[1].set_ylim([0., 1.02])
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              frameon=False, ncol=2)
    # axs[1].xaxis.set_major_formatter(date_format)
    
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[0].yaxis.set_label_position("right")
    axs[0].yaxis.tick_right()
    # axs[1].set_xlim([t[0], t[-1]])
    axs[1].grid(which='major')
    # axs[1].set_xlabel('Time (mm:ss)')
    axs[1].xaxis.get_ticklocs(minor=True)
    axs[1].yaxis.get_ticklocs(minor=True)
    axs[1].minorticks_on()
    labels = axs[1].get_xticklabels()
    # remove the first and the last labels
    labels[0] = ""
    # set these new labels
    axs[1].set_xticklabels(labels)
#     

    plt.subplots_adjust(top=0.985,
    bottom=0.1,
    left=0.0,
    right=0.945,
    hspace=0.065,
    wspace=0.2)
#     plt.show()
    output_filename = os.path.join(root_out, filename) + out_format
    plt.savefig(output_filename, transparent=False)
    plt.close(plt.gcf()) # May be better to re-write to not use plt API
# fig.autofmt_xdate()
    return output_filename

def write_audio_for_plot(text_output_filename, root, filename, output_filename, dir_out, sr):
    '''Create output audio based on input. Returns wave format. Potential for speedup for video creation by returning
    the same filetype as was input, but not implemented due to downstream processing which utilises wav files for 
    compatibility.'''
    mozz_audio_list = []
    mozz_meta = []
    has_mosquito = False
    start_time = 0
    with open(text_output_filename) as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            p = float(line[2].split()[0])
            PE = float(line[2].split()[2])
            MI = float(line[2].split()[4])

            duration = float(line[1])-float(line[0])
            
            mozz_meta.append([str(start_time), str(start_time + duration), line[0] + '-' + line[1] + '   P: ' + line[2]]) 
            
            mozz_audio_list.append(librosa.load(os.path.join(root,filename), offset=float(line[0]),
                                                     duration=duration, sr=sr)[0])
            start_time += duration  # Append length of previous prediction to transfer i
    audio_length = start_time
    audio_output_filename = os.path.join(dir_out, output_filename) + '_mozz_pred.wav'
    if mozz_audio_list:
        librosa.output.write_wav(audio_output_filename, np.hstack(mozz_audio_list), sr, norm=False)
        has_mosquito=True
    np.savetxt(audio_output_filename[:-4] + '.txt', mozz_meta, fmt='%s', delimiter='\t')
    return audio_output_filename, audio_length, has_mosquito


def write_video_for_dash(filename_image, filename_mozz_audio, mozz_audio_length, dir_out, output_filename):
    ffmpeg_command = 'ffmpeg -framerate 24 -loop 1 -y -i ' + '"' +filename_image+ '"' + ' -i ' + '"' +filename_mozz_audio+ '"'\
    + ' -t ' + str(mozz_audio_length) + ' -filter_complex \"color=c=red:s=945x4[bar];[0][bar]overlay=-(w-2)+(w/' \
    + str(mozz_audio_length) + ')*t:300:shortest=1\" \
    -c:a aac -b:a 32k -framerate 24 -vcodec libx264 -pix_fmt yuv420p  -color_primaries smpte170m -color_trc smpte170m -colorspace smpte170m -shortest ' + '"' \
    + os.path.join(dir_out, output_filename) + '_mozz_pred.mp4' + '"' 
    
    os.system(ffmpeg_command)
