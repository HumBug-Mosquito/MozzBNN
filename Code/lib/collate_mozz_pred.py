import pandas as pd
import os 

def find_mozz_pred(df_path, file_out):
    df = pd.read_csv(df_path)
    df['mozz_pred_wav'] = False
    df['mozz_pred_labels'] = False
    file_count = 0

    for idx, row in df.iterrows():
        path = row['path'].replace('/data/MozzWear', '/home/ivank/dbmount/MozzWearPlot')
        # if wave file of positive predictions exists, append the wave file and labels to dataframe:
        if os.path.isfile(path + '_mozz_pred.wav'):
            file_count+=1
            df.loc[idx, 'mozz_pred_wav'] = path + '_mozz_pred.wav'
            df.loc[idx, 'mozz_pred_labels'] = path + '_mozz_pred.txt'
    print('Found files with mosquito predictions:', file_count)
    df.to_csv(file_out, index=False)

if __name__ == '__main__':
    find_mozz_pred('/humbug-data/CDC-LT-Tanzania/CDCLTphones.csv', '/humbug-data/CDC-LT-Tanzania/CDCLTphonesMozzPred.csv')
