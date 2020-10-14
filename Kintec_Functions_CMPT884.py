# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:48:46 2018

@author: Evan Macdonald
"""
import pandas as pd
import numpy as np
import struct
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm
from sklearn.metrics import confusion_matrix
import cv2

# opens and imports data from output file from device up to Kintec_V1.1
def read_data(file_path):
    column_names = ['time', 'FSR1', 'FSR2', 'FSR3', 'FSR4', 'FSR5', 'FSR6', 'FSR7', 'X', 'Y', 'Z', 'ActivityState']
    data = pd.read_csv(
            file_path, 
            header=None, 
            skiprows=2, 
            dtype={'FSR1':np.float32, 'FSR2':np.float32, 'FSR3':np.float32, 'FSR4':np.float32, 'FSR5':np.float32, 'FSR6':np.float32, 'FSR7':np.float32, 'X':np.float32, 'Y':np.float32, 'Z':np.float32}, 
            names = column_names)
    return data

# opens data from a binary file with all float values. 
# deletes any zero values (filler bytes)
# works with Kintec_V2 teensy software
# Note: importing all data as float, have cast it all to float in the Arduino code
# future improvement would be to import as multiple datatypes to get accurate timestamps
def read_binary(file_path):
    sample = ''
    fill = ''
    for i in range(0,372):
        sample = sample + 'I10f' 
    for i in range(0,16):
        fill = fill + 'x'
    struct_fmt = sample + fill
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    results = []
    with open(file_path, 'rb') as f:
        while True:
            data_chunk = f.read(struct_len)
            if not data_chunk: break
            s = struct_unpack(data_chunk)
            results.append(s)

    raw_data = np.asarray(results)
    raw_data = np.reshape(raw_data,(-1,11))
    time_int = np.asarray(raw_data[:,0], dtype=np.dtype('u4'))

    column_names = ['time', 'FSR1', 'FSR2', 'FSR3', 'FSR4', 
                'FSR5', 'FSR6', 'FSR7', 'X', 'Y', 'Z']
    data = pd.DataFrame(raw_data, columns=column_names)
    data['time'] = time_int
    data['ActivityState'] = 0
    return data

def window(data, size, overlap):
    i = 0
    while i < data.count():
        yield int(i), int(i+size)
        i += (size/overlap) #equates to a 50% overlap of second half of data. was (size/2)


# breaks data into buffer segments       
def segment_values(data, window_size, overlap):
    numvars = 10
    segments = np.empty((0,window_size,numvars))
    labels = np.empty((0))
    for (start,end) in window(data['time'], window_size, (window_size/(window_size-overlap))):
        FSR1 = data["FSR1"][start:end]
        FSR2 = data["FSR2"][start:end]
        FSR3 = data["FSR3"][start:end]
        FSR4 = data["FSR4"][start:end]
        FSR5 = data["FSR5"][start:end]
        FSR6 = data["FSR6"][start:end]
        FSR7 = data["FSR7"][start:end]
        X = data["X"][start:end]
        Y = data["Y"][start:end]
        Z = data["Z"][start:end]
        if(len(data['time'][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([FSR1,FSR2,FSR3,FSR4,FSR5,FSR6,FSR7,X,Y,Z])])
            labels = np.append(labels, stats.mode(data['ActivityState'][start:end])[0][0])
    return segments, labels

def segment_values_2(data, window_size, overlap):
    numvars = 20
    segments = np.empty((0,window_size,numvars))
    labels = np.empty((0))
    for (start,end) in window(data['time'], window_size, (window_size/(window_size-overlap))):
        if(len(data['time'][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack(
                    [data["FSR1_R"][start:end],
                     data["FSR2_R"][start:end],
                     data["FSR3_R"][start:end],
                     data["FSR4_R"][start:end],
                     data["FSR5_R"][start:end],
                     data["FSR6_R"][start:end],
                     data["FSR7_R"][start:end],
                     data["FSR1_L"][start:end],
                     data["FSR2_L"][start:end],
                     data["FSR3_L"][start:end],
                     data["FSR4_L"][start:end],
                     data["FSR5_L"][start:end],
                     data["FSR6_L"][start:end],
                     data["FSR7_L"][start:end],
                     data["X_R"][start:end],
                     data["Y_R"][start:end],
                     data["Z_R"][start:end],
                     data["X_L"][start:end],
                     data["Y_L"][start:end],
                     data["Z_L"][start:end]])])
            labels = np.append(labels, stats.mode(data['ActivityState'][start:end])[0][0])
    return segments, labels

# Very fast way of buffering data if using no overlap in the data
def segment_values_NO(data, window_size):
    segments = np.asarray(data)
    #cut to be a shape that can fit into N*window_size
    #note, takes data off the end of the array
    maxLen = (int(len(data)/window_size))*window_size
    segments = segments[:maxLen]
    #pull out labels and crop off timestamps etc.
    labels = segments[:,22]
    segments = np.delete(segments,(0,1,22),1)
    #buffer data to the right shape
    segments = np.reshape(segments,(-1,window_size,20))
    labels = np.reshape(labels,(-1,window_size))
    labels = stats.mode(labels,axis=1)[0]
    labels = labels[:,0]
    return segments, labels

def get_features(X):
    #average values
    mean = np.mean(X,axis=2)
    #Standard Deviation
    std = np.std(X,axis=2)
    features = np.concatenate((mean,std),axis=1)
    return features    

def normalize(X):
    out = np.zeros(X.shape)
    maxFSR = np.max(X[:,:,:,0:7])
    maxACC = np.max(X[:,:,:,7:10])
    out[:,:,:,0:7]=X[:,:,:,0:7]/maxFSR
    out[:,:,:,7:10]=X[:,:,:,7:10]/maxACC
    return out

def normalize_2(X):
    out = np.zeros(X.shape)
    maxFSR = np.max(X[:,:,:,0:14])
    maxACC = np.max(X[:,:,:,14:20])
    out[:,:,:,0:14]=X[:,:,:,0:14]/maxFSR
    out[:,:,:,14:20]=np.absolute(X[:,:,:,14:20]/maxACC)
    return out

def video_analysis(video_file, solution, num_samples, save_file):
    #  Load video file
    cap = cv2.VideoCapture(video_file)

    #  Video file properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frames = int((num_samples/(1000/22))*fps) # how long the video should be in # of frames

    #  Modify solution array to fit video length
    sol_index = np.asarray(np.where(solution[:-1] != solution[1:]))
    sol_index = np.reshape(sol_index,(-1,1))
    index = sol_index*(end_frames/num_samples)
    index = index.astype(dtype=int)
    #index = np.insert(index,0,0)

    sol_vid = np.empty(end_frames,dtype=int)
    for i in range (0,len(index)-1):
        if i==0:
            sol_vid[0:index[(1,0)]] = solution[sol_index[(i,0)]-1]
        sol_vid[index[(i,0)]:index[(i+1,0)]] = solution[sol_index[(i+1,0)]-2]
    sol_vid[index[(-1,0)]:]=solution[-1]

    #  Create text version of solution array to print on video
    txt_sol = []
    for i in range(0,len(sol_vid)):
        if sol_vid[i]==1:
            txt_sol.append('SITTING')
        if sol_vid[i]==2:
            txt_sol.append('STANDING')
        if sol_vid[i]==3:
            txt_sol.append('WALKING')

    #  Play video with solutions on the screen and save file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_file, fourcc, fps, (1080,608)) 
    count=0
    while True & (count<end_frames):
        ret, frame = cap.read()
        cv2.putText(img=frame,
                    text=txt_sol[count],
                    org = (int(40), int(frameHeight-40)), 
                    fontFace = cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale = 2, 
                    color = (0, 255, 0))
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(int(1000/fps)-5) & 0xFF == ord('q'): #this is where you can change the framerate
            break
        count = count + 1

    cap.release()
    cv2.destroyAllWindows() 
