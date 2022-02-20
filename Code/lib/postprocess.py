import os
import sys
sys.path.insert(0, os.path.abspath('../lib/VAD-Humbug'))
sys.path.insert(0, os.path.abspath('../lib'))
import os
# if not os.getcwd().endswith('VAD-Humbug'):
#   os.chdir('../lib/VAD-Humbug')
from VAD_pipeline import VAD_pipeline
import matplotlib.pyplot as plt

import pickle
import util
import librosa
import numpy as np
import soundfile as sf
import ipdb



# # Postprocessing

# In[17]:


import csv


# In[5]:


# Assumes format:
# ['458.24', '460.8', '0.56 PE: 0.99 MI: 0.12']

# Can turn into class which supports method for BNN, method for RF (based on different label formats)

def get_audio_detected(rootFolderPath, accept_list, audio_format, sr, p_threshold, PE_threshold, MI_threshold):
    mozz_audio_list = []
    for root, dirs, files in os.walk(rootFolderPath):
        for filename in files:
            if filename.endswith('.txt'):
                for accept_item in accept_list:
                    if accept_item in filename:
                        print('accepted file:', filename)
                        with open(os.path.join(root, filename)) as f:
                            reader = csv.reader(f, delimiter='\t')
                            for line in reader:
                                p = float(line[2].split()[0])
                                PE = float(line[2].split()[2])
                                MI = float(line[2].split()[4])

                                if p > p_threshold and PE < PE_threshold and MI < MI_threshold:
                                    mozz_audio_list.append(librosa.load(os.path.join(root,filename.partition(audio_format)[0] +
                                                                                 filename.partition(audio_format)[1]), offset=float(line[0]),
                                                                             duration=float(line[1])-float(line[0]), sr=sr)[0])
    return mozz_audio_list


# In[14]:


def get_audio_detected_meta(rootFolderPath, accept_list, model_name, audio_format, sr, p_threshold, PE_threshold, MI_threshold):
    mozz_audio_list = []
    mozz_meta = []
    start_time = 0
    counter=0
    for root, dirs, files in os.walk(rootFolderPath):
        for filename in files:
            if filename.endswith(model_name + '.txt'):
                for accept_item in accept_list:
                    if accept_item in filename:
                        with open(os.path.join(root, filename)) as f:
                            counter+=1
                            print('Processing file number:', counter, filename, rootFolderPath)
                            reader = csv.reader(f, delimiter='\t')
                            for line in reader:
                                p = float(line[2].split()[0])
                                PE = float(line[2].split()[2])
                                MI = float(line[2].split()[4])

                                if p > p_threshold and PE < PE_threshold and MI < MI_threshold:
                                    duration = float(line[1])-float(line[0])
                                    
                                    mozz_meta.append([str(start_time), str(start_time + duration), filename.partition(audio_format)[0] + "  " + "{:.2f}".format(float(line[0])) + "-" +
                                                      "{:.2f}".format(float(line[1]))]) 
                                    mozz_audio_list.append(librosa.load(os.path.join(root,filename.partition(audio_format)[0] +
                                                                                 filename.partition(audio_format)[1]), offset=float(line[0]),
                                                                             duration=duration, sr=sr)[0])
                                    start_time += duration  # Append length of previous prediction to transfer into concatenated audio
    return mozz_audio_list, mozz_meta


# In[7]:


def return_file_paths(rootFolderPath, accept_list):
    filenames = []
    duration = 0
    for root, dirs, files in os.walk(rootFolderPath):
        for filename in files:
            if filename.endswith('.wav'):
                    for accept_item in accept_list:
                        if accept_item in filename:
#                 print(root[112:], filename)
                            audio_length = librosa.get_duration(filename=os.path.join(root, filename))
                            duration+= audio_length
                            filenames.append([root[112:] + "\\" + filename, audio_length])
    return filenames, duration
    


# In[218]:


rootFolderPath = 'D:/Postdoc/Data/OneDrive_2020-11-30/IFAKARA - NOVEMBER 2020 BEDNET PHONE DOWNLOADS/Semi-field raw data Nov 2020/'
phones = ['Chamber A phone 1 (A1)', 'Chamber A phone 2 (A2)', 'Chamber A phone 3 (A3)', 'Chamber A phone 4 (A4)',
         'Chamber B phone 1 (B1)', 'Chamber B phone 2 (B2)', 'Chamber B phone 3 (B3)', 'Chamber B phone 4 (B4)',
         'Chamber C phone 1 (C1)', 'Chamber C phone 2 (C2)', 'Chamber C phone 3 (C3)', 'Chamber C phone 4 (C4)',
         'Chamber D phone 1 (D1)', 'Chamber D phone 2 (D2)', 'Chamber D phone 3 (D3)', 'Chamber D phone 4 (D4)']

accept_dates = ['2020-11-16', '2020-11-17', '2020-11-18', '2020-11-19', '2020-11-20', '2020-11-21', '2020-11-22', '2020-11-23', 
                '2020-11-24', '2020-11-25', '2020-11-26', '2020-11-27']
duration_list = []
for phone in phones:
    filenames, duration = return_file_paths(rootFolderPath+phone, accept_dates)
    np.savetxt("PredictionPathsDuration16th27th" + phone + ".csv", filenames, delimiter=",", fmt='%s')
    duration_list.append([phone, duration])
np.savetxt("AudioDuration16th27th.csv", duration_list, delimiter=",", fmt='%s')


# In[8]:


import IPython.display as ipd


# In[8]:


for phone in phones:
    print('meh')


# In[11]:


p_threshold = 0.5
PE_threshold = 1.0
MI_threshold = 0.1
# '../audio_out/2020-11-16_to_2020-11-27_' + 'P' + str(p_threshold) + '_' + 'PE' + str(PE_threshold) + '_' + 'MI' + str(MI_threshold) + '/all' + '.txt'


# In[88]:


### Prototype code for getting wave files from dataframe of phone UUIDs: 06/10/2021:
import pandas as pd
CDC_phones = pd.read_csv('F:\PostdocData\HumBugServer\CDC-LT\CDCLTphones.csv')


# In[89]:


CDC_phones['mozz_pred_wav'] = False
CDC_phones['mozz_pred_labels'] = False


# In[80]:


CDC_phones


# In[ ]:


df.loc[df['A'] > 2, 'B'] = new_val


# In[90]:


CDC_phones.loc[0, 'mozz_pred_wav'] = 'Sammy'


# In[91]:


CDC_phones


# In[58]:


# Latest file on server, push to git!


# In[59]:


find_mozz_pred(CDC_phones,'F:\PostdocData\HumBugServer\CDC-LT\CDCLTphonesMozzPred.csv')


# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv('F:\PostdocData\HumBugServer\CDC-LT\CDCLTphonesMozzPred.csv')


# In[106]:


row['mozz_pred_wav']


# In[123]:


'2021-08-17T19:08:08.158Z'[:10]


# In[119]:


from datetime import datetime


# In[124]:


datetime.strptime(row['datetime_recorded'][:10],'%Y-%m-%d')


# In[126]:


lines = []

for idx, row in df.iterrows():
    if row['mozz_pred_wav'] != 'False':
        if datetime.strptime(row['datetime_recorded'][:10],'%Y-%m-%d') > datetime(2021, 3, 17, 0, 0):
            lines.append(row['mozz_pred_wav'] + '\n')
            lines.append(row['mozz_pred_labels'] + '\n')
            
with open('CDC-LT-mozz_pred_files.txt', 'w') as f:
    f.writelines(lines)


# In[189]:


file.partition('_mozz_pred.txt')[0]


# In[222]:


df[df['mozz_pred_wav'].str.contains('r2025-06-09_11.23.58.957__v9.aac_u2021-08-14_17.12.45.066997_mozz_pred.wav')].uuid


# In[210]:


df['mozz_pred_wav'].iloc[1].split('/')[:-1]


# In[197]:


file


# In[230]:


df_match


# In[233]:


# Get list of all UUIDs:

pd.unique(df.uuid)

for uuid in pd.unique(df.uuid):
    df_match = df[df.uuid == uuid].mozz_pred_labels
    for file in df_match:
        if file != 'False':
            print(file.split('/')[-1])


# In[11]:


import csv
root = 'F:/PostdocData/HumBugServer/CDC-LT/bnn_neurips_best_mozz_pred/'

# root = 'G:/CDC-LT-Tanzania/'

p_threshold = 0.5
MI_threshold = 1.0


# Get list of all UUIDs:

pd.unique(df.uuid)

for uuid in pd.unique(df.uuid):
    mozz_meta = []
    sig_list = []
    total_time = 0
    t_start_out = 0 
    df_match = df[df.uuid == uuid].mozz_pred_labels
    for filename in df_match:
        if filename != 'False':
            file = filename.split('/')[-1]
            with open(os.path.join(root,file)) as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    t_start = float(line[0])
                    t_end = float(line[1])
                    t_start_end_original = line[2].split()[0]
                    p = float(line[2].split()[2])
                    PE = float(line[2].split()[4])
                    MI = float(line[2].split()[6])
                    if ((p > p_threshold) & (MI < MI_threshold)):
    #                     print('Accepted:', file)
                        t_end_out = t_start_out + (t_end-t_start)              
                        mozz_meta.append([str(t_start_out), str(t_end_out), file.partition('_mozz_pred.txt')[0] + "  " + t_start_end_original]) 
                        sig, rate = librosa.load(root+file.replace('.txt','.wav'), sr=None, offset=t_start, duration =t_end-t_start)
                        total_time += len(sig)/rate
                        sig_list.append(sig)
                        t_start_out = t_end_out
    np.savetxt('G:/CDC-LT-Tanzania/phones5/'+ 'uuid_' + str(uuid) + '_P'+str(p_threshold)+'_'+'MI'+str(MI_threshold)+ '.txt',
           mozz_meta, fmt='%s', delimiter='\t')
    librosa.output.write_wav('G:/CDC-LT-Tanzania/phones5/'+'uuid_' + str(uuid) + '_P'+str(p_threshold)+'_'+'MI'+str(MI_threshold)+ '.wav',
                         np.hstack(sig_list), rate, norm=False)
    print('Total length of audio in seconds for uuid:', total_time, uuid)                   


# In[193]:


116983/60/60


# In[196]:


np.savetxt('F:/PostdocData/HumBugServer/CDC-LT/BNN_neurips_Best/'+'P'+str(p_threshold)+'_'+'MI'+str(MI_threshold)+ '.txt',
           mozz_meta, fmt='%s', delimiter='\t')
librosa.output.write_wav('F:/PostdocData/HumBugServer/CDC-LT/BNN_neurips_Best/'+'P'+str(p_threshold)+'_'+'MI'+str(MI_threshold)+ '.wav',
                         np.hstack(sig_list), rate, norm=False)


# In[181]:


mozz_meta


# In[143]:


line[0]
line[1]
line[2]


# In[148]:


line[2].split()[4]


# In[93]:


# phones = ['Chamber A_phone 1 (A1)', 'Chamber A_phone 2 (A2)', 'Chamber A_phone 3 (A3)', 'Chamber A_phone 4 (A4)',
#          'Chamber B_phone 1 (B1)', 'Chamber B_phone 2 (B2)', 'Chamber B_phone 3 (B3)', 'Chamber B_phone 4 (B4)',
#          'Chamber C_phone 1 (C1)', 'Chamber C_phone 2 (C2)', 'Chamber C_phone 3 (C3)', 'Chamber C_phone 4 (C4)',
#          'Chamber D_phone 1 (D1)', 'Chamber D_phone 2 (D2)', 'Chamber D_phone 3 (D3)', 'Chamber D_phone 4 (D4)']


phones = ['']
# phones = ['Chamber D_phone 1 (D1)']
# accept_dates = ['2021-02-', '2021-03-']


# Old experiment
accept_dates = ['2020-11-16', '2020-11-17', '2020-11-18', '2020-11-19', '2020-11-20', '2020-11-21', '2020-11-22', '2020-11-23', 
                '2020-11-24', '2020-11-25', '2020-11-26', '2020-11-27']


# accept_dates =  ['2020-11-09', '2020-11-10', '2020-11-11', '2020-11-12', '2020-11-13']

model_name_step = 'step_30_samples_10_' + model_name

audio_format = '.wav'
sr = 8000
p_threshold = 0.8
PE_threshold = 0.5
MI_threshold = 0.09
for phone in phones:
    rootFolderPath = 'F:/PostdocData/HumBugServer/SemiFieldDataTanzania/' + phone    
    mozz_audio_list, mozz_meta = get_audio_detected_meta(rootFolderPath, accept_dates, model_name_step, audio_format, sr, p_threshold, PE_threshold, MI_threshold)
    if mozz_audio_list:
        np.savetxt('../audio_out/Tanzania2020/BNN_neurips_Best/2020-11-16_to_2020-11-27_0_8_0_5_0_09/2020_'+'P'+str(p_threshold)+'_'+'PE'+str(PE_threshold)+'_'+'MI'+str(MI_threshold)+ '_step_' + str(step_size) + phone +'.txt',
                   mozz_meta, fmt='%s', delimiter='\t')
    # ipd.Audio(np.hstack(mozz_audio_list), rate=sr)
        librosa.output.write_wav('../audio_out/Tanzania2020/BNN_neurips_Best/2020-11-16_to_2020-11-27_0_8_0_5_0_09/2020_'+'P'+str(p_threshold)+'_'+'PE'+str(PE_threshold)+'_'+'MI'+str(MI_threshold)+'_step_' + str(step_size) + phone + '.wav', np.hstack(mozz_audio_list), sr, norm=False)


# 3517 files for  accept_dates = ['2020-11-16', '2020-11-17', '2020-11-18', '2020-11-19', '2020-11-20', '2020-11-21', '2020-11-22', '2020-11-23', 
#                 '2020-11-24', '2020-11-25', '2020-11-26', '2020-11-27']

# In[53]:

