# imports
import numpy as np
import pandas as pd
import heartbeat_extraction as he
import feature_creation as fc
import matplotlib.pyplot as plt
import ANN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#----------- IMPORT RAW ECG SIGNALS -----------#
"""
x_train_ = pd.read_csv('X_train.csv').drop('id', axis=1)
y_train = pd.read_csv('y_train.csv').drop('id', axis=1)
"""

#----------- EXTRACT HEARTBEAT SIGNALS -----------#
"""
signals = he.biosppy(sampling_rate=300)
y_train1 = signals.fit(x_train_,y_train.to_numpy(),show=False)
signals.toCSV(y_train1)
"""


#----------- IMPORT EXTRACTED HEARTBEATS FROM CSV -----------#

mean_train = pd.read_csv('mean')
std_train = pd.read_csv('std')
y_train1 = pd.read_csv('y_train1')

#----------- CREATE FEATURES FROM HEARTBEATS -----------#

features = fc.feature_creation()
features.createFeatures(std_train.to_numpy(),mean_train.to_numpy())
df = features.getFeatures()

#----------- PLOT FEATURES -----------#

features.plotMeanStd(y_train1,mean_train,std_train)

#----------- TRAIN ANN CLASSIFYING 0,1,2 FROM 3 -----------#

y_train_combined = y_train1.to_numpy()
y_train_combined[y_train_combined == 1] = 0
y_train_combined[y_train_combined == 2] = 0
y_train_combined[y_train_combined == 3] = 1

x_train, x_test_val, y_train, y_test_val = train_test_split(df.to_numpy(), y_train_combined, test_size=0.4, random_state=42)
ANN.fit_MLP1(x_train,x_test_val,np.squeeze(y_train),np.squeeze(y_test_val))

#----------- TRAIN SVC CLASSIFYING 0,1,2 FROM 3 -----------#


#----------- TRAIN KNN CLASSIFYING 0,1,2 FROM 3 -----------#

#----------- COMBINE ALL CLASSIFIERS ---------#