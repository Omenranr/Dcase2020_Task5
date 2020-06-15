import librosa
import numpy as np
import sys,os
import pickle
from tqdm import tqdm
import os
import sys
from multiprocessing import Pool, Process, Queue
import efficientnet.keras as efn
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("input_path")
parser.add_argument("output_path")
args = parser.parse_args()


output_path = args.output_path
input_path = args.input_path

basic_sample = 48000 ######다시시작
resampling = 44100 ###### 22050 --> 1104   32000 --> 800   44100*2206    BEST 44100 1024
n_fft = 1024
win_length = 1024
hop_length = 512

n_mels = 128
PROCESSOR = 16
mono = True
RESAMPLE = 'kaiser_best'
extraction = True #False

second = 5

def feature_extracion(name, srate=resampling):
    name = name.split('?')
    filename = name[0]
    second = int(name[3])
    outpath = name[4]

    print("\tProgress : " , str((int(int(name[1])/int(name[2])*100))), "%")

    try:
        y, sr = librosa.load(filename, sr=basic_sample, mono=mono)
    except:
        raise IOError('Give me an audio  file which I can read!!')

    if sr != srate:
        y = librosa.resample(y,sr,srate, res_type=RESAMPLE)  ### 1 fast 2 best 3 fft 4 scpy

    time_data = []
    feature_data= []

    ### 5S EfficientNet feature
    try:
        for i in range(2):

            time_data.append(i)

            i = i * resampling * second

            if i + int(resampling*second) > resampling*10:
                break

            split_y =y[i : i + int(resampling*second)]

            sp_harmonic, sp_percussive = librosa.effects.hpss(split_y)
            sp_y = librosa.util.normalize(split_y, norm=np.inf, axis=None)
            sp_harmonic = librosa.util.normalize(sp_harmonic, norm=np.inf, axis=None)
            sp_percussive = librosa.util.normalize(sp_percussive, norm=np.inf, axis=None)

            inpt = librosa.power_to_db(librosa.feature.melspectrogram(y=sp_y,sr=srate,n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels = n_mels))
            inpt2 = librosa.power_to_db(librosa.feature.melspectrogram(y=sp_harmonic,sr=srate,n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels = n_mels))
            inpt3 = librosa.power_to_db(librosa.feature.melspectrogram(y=sp_percussive,sr=srate,n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels = n_mels))

            stack_inpt = np.stack(([inpt], [inpt2], [inpt3]), axis=3)

            stack_inpt = librosa.util.normalize(stack_inpt, norm=np.inf, axis=None)

            feature = np.reshape(stack_inpt,(stack_inpt.shape[1],stack_inpt.shape[2],stack_inpt.shape[3]))
            feature_data.append(feature)


    except Exception as e:
        print(e)

    filename = filename.split('/')[-1]
    time_data = np.asarray(time_data)
    feature_data = np.asarray(feature_data)
    print(np.shape(feature_data))
    print(outpath + filename[:-4] + '.npz')
    np.savez(outpath + filename[:-4], timestamps=time_data, embedding=feature_data)

    return feature_data

if __name__ == '__main__':

    if extraction :
        train_folder = sorted(os.listdir(input_path))
        print(train_folder[:5])
        print("Data Feature Extraction")

        p = Pool(processes=PROCESSOR)
        data = p.map(feature_extracion, [input_path + '/' + train_folder[j] + '?' + str(j) + '?' + str(len(train_folder)) +'?' + str(second) + '?' + str(output_path) for j in range(len(train_folder))], chunksize=1)
        p.close()
