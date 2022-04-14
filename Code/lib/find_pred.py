from pymongo import MongoClient
import datetime
import pandas as pd
import os
import librosa
import numpy as np
import csv

# Parameters to be turned into function arguments, or parsed from dataframe/csv:
# small sample from Marianne's uuid list (email March 02 11:35AM)

#phase 1 (Bandundu) - 03/11/21 to 20/11-21
#phase 2 (Kinshasa) - 19/12/21 to 07/01/22

start_datetime = datetime.datetime(2021, 11, 3, 0, 0, 0, 0)
end_datetime = datetime.datetime(2021, 11, 20, 23, 59, 59, 999999)

uuid_list = ['8a927dd5e08dbef5',
'10fbf4b13c671633',
'c3d4fee59c9c1d61',
'5a610a26bbad4f14',
'4d3267fd607d95aa',
'4466eac2bc69b460',
'45db7eb36c7ff23b',
'b1d6b22d0c4ce91f',
'15830e8d271a6cdb',
'f7ddba7798a1c699',
'34e297039f74b322',
'99c4599e66ea233e',
'94062f72a15b1a4b',
'f4afabece3523bb3',
'4ed245c3a206ea0c',
'050aab6fc4651530',
'dec13370cfa75e87',
'f66362ad68730ec6',
'9cb110903f13bc00',
'e8f4fb5267a1527c',
'5b877b5a4a47c1fc',
'484702488240b8bb',
'fc23d8eb1dea2de2',
'13b41fda42f8fa85',
'03d783523b494250',
'c55f8f34cf2e361f',
'0c3da0a3ca8e17c4',
'f233297f6aa1c68a',
'57f6be3e2678ff2d',
'6a0db05c070473fe']



# For post-processing and saving results:

# root = '/home/ivank/dbmount/MozzWearPlot/'
root = ''  # Check the location of filepath root is pointing to: currently superfluous parameter, but label OUTPUT could be shortened (see comments in code)
out_dir = '/home/ivank/audio_out/'
# root = 'G:/CDC-LT-Tanzania/'

p_threshold = 0.8  # probability threshold over which to accept positive predictions as mosquito
MI_threshold = 1.0 # uncertainty threshold UNDER which to accept positive predictions as mosquito (1.0 is maximum entropy, see paper for parameter settings)



client = MongoClient('mongodb://humbug.ac.uk/')
db = client['backend_upload']

''' Example field:
 db['reports'].find_one()
 returns:
{'_id': ObjectId('55cb4efa7cdf33532641047d'), 'uuid': 'd323c5c00fd24998', 'datetime_received': datetime.datetime(2015, 8, 12, 14, 49, 46, 982000),
 'path': '/data/MozzWear/2015-08-10/19/r2015-08-10_19.15.29.wav_u2015-08-12_14.49.46.982279.wav', 'model': 'ONE TOUCH 4015X',
  'datetime_recorded': datetime.datetime(2015, 8, 10, 19, 15, 29), 'manufacturer': 'Orange'}
'''

recordings = db['reports']
# Return query for a certain uuid, with date recorded after ("$gt: greater than") datetime.datetime object


             



myquery = {"uuid": {"$in": uuid_list}, "datetime_recorded": {"$gt": start_datetime, "$lt": end_datetime}}
mydoc = recordings.find(myquery)

df = pd.DataFrame(list(mydoc))



# pasted from collate_mozz_pred.py

def find_mozz_pred(df, file_out):
    df['mozz_pred_wav'] = 'False'
    df['mozz_pred_labels'] = 'False'
    file_count = 0

    for idx, row in df.iterrows():
        path = row['path'].replace('/data/MozzWear', '/home/ivank/dbmount/MozzWearPlot')
        # Warning: some files are missing an audio extension, whereas others have it present. In newer versions, the syntax below works:
        # print(path[:-4] + '_mozz_pred.wav'). In older, path[:] + '_mozz_pred.wav'
        # if wave file of positive predictions exists, append the wave file and labels to dataframe:
        if os.path.isfile(path[:-4] + '_mozz_pred.wav'):
            file_count+=1
            df.loc[idx, 'mozz_pred_wav'] = path[:-4] + '_mozz_pred.wav'
            df.loc[idx, 'mozz_pred_labels'] = path[:-4] + '_mozz_pred.txt'
    print('Found files with mosquito predictions:', file_count)
    df.to_csv(file_out, index=False)


# Intermediate output to 
find_mozz_pred(df,'../../MongoDBqueries/uuid_out.csv')

# Code below assumed reading dataframe, type of Boolean instead entered as string for entries in df['mozz_pred_wav'] (can be streamlined)
df = pd.read_csv('../../MongoDBqueries/uuid_out.csv',dtype=str)
lines = []

for idx, row in df.iterrows():
    if row['mozz_pred_wav'] != 'False':
        lines.append(row['mozz_pred_wav'] + '\n')
        lines.append(row['mozz_pred_labels'] + '\n')

with open('debug_mozz_pred_files.txt', 'w') as f:
    f.writelines(lines)



# Loop through UUID, and find all outputs from predict.py to parse in new format.
#TODO:
# For corresponding existing predictions (mozz_pred_wav), we should also look for (mozz_pred_species), which is to be run on just the output (?)
# Currently code only outputs MED (positive mosquito events with no species recognition)
# Get list of all UUIDs (legacy code upon loading dataframe): (Could load directly from input/check output as expected)
for uuid in pd.unique(df.uuid):
    mozz_meta = []
    sig_list = []
    total_time = 0
    t_start_out = 0
    df_match = df[df.uuid == uuid].mozz_pred_labels # Find all records for uuid where positive prediction exists
    for filename in df_match:
        if filename != 'False':
            file = filename #.split('/')[-1] # Make sure extensions are read properly
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
    # Example output line for mozz_meta. Consult with Zoology whether best to shorten to just filename (not full path)
    # 1561.0239999999994      1562.9439999999995      /home/ivank/dbmount/MozzWearPlot/2021-11-08/4/r2021-11-08_04.04.42.776__v543.aac_u2021-11-08_16.53.43.524070  53.76-55.68
                        sig, rate = librosa.load(root+file.replace('.txt','.wav'), sr=None, offset=t_start, duration =t_end-t_start)
                        total_time += len(sig)/rate
                        sig_list.append(sig)
                        t_start_out = t_end_out
    np.savetxt(out_dir+ 'uuid_' + str(uuid) + '_P'+str(p_threshold)+'_'+'MI'+str(MI_threshold)+ '.txt',
           mozz_meta, fmt='%s', delimiter='\t')
    librosa.output.write_wav(out_dir+'uuid_' + str(uuid) + '_P'+str(p_threshold)+'_'+'MI'+str(MI_threshold)+ '.wav',
                         np.hstack(sig_list), rate, norm=False)
    print('Total length of audio in seconds for uuid:', total_time, uuid)
