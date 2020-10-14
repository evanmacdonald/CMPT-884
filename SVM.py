# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:00:08 2018

@author: Evan Macdonald

Trains SVM classification algorithm using train data and then outputs results from test data
"""

#Imports
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import svm
import time
from sklearn.metrics import confusion_matrix
from Kintec_Functions_CMPT884 import segment_values_NO
from Kintec_Functions_CMPT884 import get_features
from Kintec_Functions_CMPT884 import normalize_2
from data_assembly_two_insole_CMPT884 import combined_data_two_insole


#%% import all data
print('Importing Data...')
train_data, test_data = combined_data_two_insole()
#test_video_file = 'C:/Users/Evan Macdonald/sfuvault/Thesis/Trial Data/Two_device/Videos/two_device_train_edit_2.mp4'
print('Data imported...')

#%% This is now the slowest part..

print('Buffering Data...')
# buffer data
buffLen = 135 #45, 90, 135, 180, 225
numvars = 20
overlap = buffLen/4 # Number of datapoints to overlap in the buffers

train_segments, train_labels = segment_values_NO(train_data,buffLen)
train_y = np.asarray(pd.get_dummies(train_labels), dtype = np.int8)
train_x = train_segments.reshape(len(train_segments), 1, buffLen, numvars)

test_segments, test_labels = segment_values_NO(test_data,buffLen)
test_y = np.asarray(pd.get_dummies(test_labels), dtype = np.int8)
test_x = test_segments.reshape(len(test_segments), 1, buffLen, numvars)
print('Data buffered...')

#%% train SVM

X = get_features(normalize_2(train_x))
X = np.reshape(X,[-1,X.shape[1]*X.shape[2]])
Y = train_labels

model = svm.SVC(C=4, kernel='sigmoid', gamma='auto', max_iter=10000)
model.fit(X,Y)
print('Model has been fitted...')

#start timer
start_time = time.time()

# get predictions for test set
T = get_features(normalize_2(test_x))
T = np.reshape(T,[-1,T.shape[1]*T.shape[2]])
y_ = model.predict(T)

#end timer
end_time = time.time()
time_dif = end_time - start_time

#%%  reshape y_ to match test_data
y_index = np.asarray(np.where(y_[:-1] != y_[1:]))
y_index = np.reshape(y_index,(-1,1))
index = y_index*(len(test_data)/len(y_))
index = index.astype(dtype=int)

y_out_val = np.empty(len(test_data),dtype=int)
for i in range (0,len(index)-1):
    if i==0:
        y_out_val[0:index[(1,0)]] = y_[y_index[(i,0)]-1]
    y_out_val[index[(i,0)]:index[(i+1,0)]] = y_[y_index[(i+1,0)]-2]
y_out_val[index[(-1,0)]:]=y_[-1]

#%%  reshape test_labels to match y_out_val
yt_index = np.asarray(np.where(test_labels[:-1] != test_labels[1:]))
yt_index = np.reshape(yt_index,(-1,1))
index = yt_index*(len(test_data)/len(test_labels))
index = index.astype(dtype=int)

## use these three lines if using data with only two chunks of data (i.e. sitting and then standing)
#y_test_val = np.empty(len(test_data),dtype=int)
#y_test_val[0:index[0,0]] = test_labels[0]
#y_test_val[index[0,0]:] = test_labels[int(yt_index[0,0]+1)]

# use this block for anything >2 chunks of data
y_test_val = np.empty(len(test_data),dtype=int)
for i in range (0,len(index)-1):
    if i==0:
        y_test_val[0:index[(1,0)]] = test_labels[yt_index[(i,0)]-1]
    y_test_val[index[(i,0)]:index[(i+1,0)]] = test_labels[yt_index[(i+1,0)]-2]
y_test_val[index[(-1,0)]:]=test_labels[-1]

#%% confusion matrix and stats
# comment this section out if there are not labels for test data
#input("Results are ready, press Enter to continue...")

cm = confusion_matrix(test_labels, y_)
plt.figure(2)
sn.heatmap(cm, annot=True, fmt='g',annot_kws={"size": 30}, 
           square=True, xticklabels=['Sit','Stand','Walk'],
           yticklabels=['Sit','Stand','Walk'])

stat = np.zeros([3,4])
for i in range(0,3):
    stat[i,0] = cm[i,i]
    stat[i,1] = (cm[0,0]+cm[1,1]+cm[2,2])-cm[i,i]
    stat[i,2] = np.sum(cm[:,i])-cm[i,i]
    stat[i,3] = np.sum(cm[i,:])-cm[i,i]

stat = np.mean(stat,axis=0)
TP = stat[0]
TN = stat[1]
FP = stat[2]
FN = stat[3]

#accuracy = (TP+TN)/(TP+TN+FP+FN) #(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm)
accuracy = (np.trace(cm))/sum(sum(cm))
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F = 2*(precision*recall)/(precision+recall) #something is fucked up about this

#print('Accuracy: ', (accuracy*100), 'F measure: ', (F*100))
print('Accuracy: ', (accuracy*100), '(',(np.trace(cm)),'/',sum(sum(cm)),')')
msg = "Classification time usage: {0:>6.5} seconds"
print(msg.format(time_dif))
print(cm)

#%%plot results

fig,ax=plt.subplots(sharex=True, nrows=3, ncols=1)
ax[0].plot(test_data['FSR1_R'], linewidth=0.5)
ax[0].plot(test_data['FSR2_R'], linewidth=0.5)
ax[0].plot(test_data['FSR3_R'], linewidth=0.5)
ax[0].plot(test_data['FSR4_R'], linewidth=0.5)
ax[0].plot(test_data['FSR5_R'], linewidth=0.5)
ax[0].plot(test_data['FSR6_R'], linewidth=0.5)
ax[0].plot(test_data['FSR7_R'], linewidth=0.5)
ax[0].set_ylim((0,900))
ax[0].legend()
ax[0].set_title('FSR data Right')
ax[0].set_ylabel('FSR Output')

ax[1].plot(test_data['FSR1_L'], linewidth=0.5)
ax[1].plot(test_data['FSR2_L'], linewidth=0.5)
ax[1].plot(test_data['FSR3_L'], linewidth=0.5)
ax[1].plot(test_data['FSR4_L'], linewidth=0.5)
ax[1].plot(test_data['FSR5_L'], linewidth=0.5)
ax[1].plot(test_data['FSR6_L'], linewidth=0.5)
ax[1].plot(test_data['FSR7_L'], linewidth=0.5)
ax[1].set_ylim((0,900))
ax[1].legend()
ax[1].set_title('FSR Data Left')
ax[1].set_ylabel('FSR Output')

ax[2].plot(test_data['ActivityState'], label='Actual')
ax[2].plot(y_out_val, label='Predicted')
ax[2].set_ylim((0.5,3.5))
ax[2].legend()
ax[2].set_title('Activity State. F-Measure = %1.3f' %F)
ax[2].set_xlabel('Sample Number')
ax[2].set_ylabel('1=Sit, 2=Stand, 3=Walk')

mng=plt.get_current_fig_manager() 
mng.window.showMaximized() #maximize figure 
plt.show()

#%%
y_out_val_pd = pd.DataFrame(y_out_val)
SID = 'SID15'
Day = 'Solutions_SVM'

y_out_val_pd.to_csv('Data/' + SID + '_' + Day + '.csv')