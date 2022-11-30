# imports
import numpy as np
import pandas as pd
import heartbeat_extraction as he
import feature_creation as fc
import matplotlib.pyplot as plt

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
features.createFeatures(std_train,mean_train)
df = features.getFeatures()

#----------- PLOT FEATURES -----------#

features.plotMeanStd(y_train1)

#----------- TRAIN SVM MEAN ON STANDART DEVIATIONN-----------#

