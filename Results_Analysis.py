# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 11:12:58 2018

@author: evanm
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

#Training data    
file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID04/SynchronizedData/SID04_Calibration.csv')
SID04 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID05/SynchronizedData/SID05_Calibration.csv')
SID05 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID06/SynchronizedData/SID06_Calibration.csv')
SID06 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID07/SynchronizedData/SID07_Calibration.csv')
SID07 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID08/SynchronizedData/SID08_Calibration.csv')
SID08 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID09/SynchronizedData/SID09_Calibration.csv')
SID09 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID10/SynchronizedData/SID10_Calibration.csv')
SID10 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID11/SynchronizedData/SID11_Calibration.csv')
SID11 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID12/SynchronizedData/SID12_Calibration.csv')
SID12 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID13/SynchronizedData/SID13_Calibration.csv')
SID13 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID14/SynchronizedData/SID14_Calibration.csv')
SID14 = pd.read_csv(file)

file = open('C:/Users/evanm/sfuvault/Thesis/Trial Data/Full Trial/SID15/SynchronizedData/SID15_Calibration.csv')
SID15 = pd.read_csv(file)

#CNN Results
train_data = pd.concat([SID04,SID05,SID06,SID07,SID08,SID09,SID10,SID11,SID12,SID13,SID14,SID15], axis=0)
train_data = train_data.reset_index(drop=True)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID04_Solutions_CNN_0005.csv')
SID04 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID05_Solutions_CNN.csv')
SID05 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID06_Solutions_CNN.csv')
SID06 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID07_Solutions_CNN.csv')
SID07 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID08_Solutions_CNN.csv')
SID08 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID09_Solutions_CNN_0005.csv')
SID09 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID10_Solutions_CNN.csv')
SID10 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID11_Solutions_CNN.csv')
SID11 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID12_Solutions_CNN.csv')
SID12 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID13_Solutions_CNN.csv')
SID13 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID14_Solutions_CNN_0005.csv')
SID14 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID15_Solutions_CNN_0005.csv')
SID15 = pd.read_csv(file)

test_data = pd.concat([SID04,SID05,SID06,SID07,SID08,SID09,SID10,SID11,SID12,SID13,SID14,SID15], axis=0)
test_data_CNN = test_data.reset_index(drop=True)

#SVM Results
file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID04_Solutions_SVM.csv')
SID04 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID05_Solutions_SVM.csv')
SID05 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID06_Solutions_SVM.csv')
SID06 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID07_Solutions_SVM.csv')
SID07 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID08_Solutions_SVM.csv')
SID08 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID09_Solutions_SVM.csv')
SID09 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID10_Solutions_SVM.csv')
SID10 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID11_Solutions_SVM.csv')
SID11 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID12_Solutions_SVM.csv')
SID12 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID13_Solutions_SVM.csv')
SID13 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID14_Solutions_SVM.csv')
SID14 = pd.read_csv(file)

file = open('C:/Users/evanm/OneDrive/SFU/2018/CMPT 884/Paper/Results/SID15_Solutions_SVM.csv')
SID15 = pd.read_csv(file)

test_data = pd.concat([SID04,SID05,SID06,SID07,SID08,SID09,SID10,SID11,SID12,SID13,SID14,SID15], axis=0)
test_data_SVM = test_data.reset_index(drop=True)


#%%

cm_CNN = confusion_matrix(train_data['ActivityState'],test_data_CNN['0'])

accuracy_CNN = (np.trace(cm_CNN))/sum(sum(cm_CNN))

print(accuracy_CNN)
print(cm_CNN)

#%%

cm_SVM = confusion_matrix(train_data['ActivityState'],test_data_SVM['0'])

accuracy_SVM = (np.trace(cm_SVM))/sum(sum(cm_SVM))

print(accuracy_SVM)
print(cm_SVM)




