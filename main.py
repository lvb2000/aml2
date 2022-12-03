# imports
import numpy as np
import pandas as pd
import heartbeat_extraction as he
import feature_creation as fc
import matplotlib.pyplot as plt
import ANN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
"""
mean_train = pd.read_csv('mean')
std_train = pd.read_csv('std')
y_train1 = pd.read_csv('y_train1')
"""
#----------- CREATE FEATURES FROM HEARTBEATS -----------#
"""
features = fc.feature_creation()
features.createFeatures(std_train.to_numpy(),mean_train.to_numpy())
df = features.getFeatures()
"""
#----------- PLOT FEATURES -----------#
"""
features.plotMeanStd(y_train1,mean_train,std_train)
"""

#----------- IMPORT FEATURES -----------#

x_train = pd.read_csv('features_ata.csv')
x_test = pd.read_csv('test_features_ata.csv')
y_train = pd.read_csv('y_train.csv')
x_train.drop(x_train.columns[0],axis=1,inplace=True)
x_test.drop(x_test.columns[0],axis=1,inplace=True)
y_train.drop(y_train.columns[0],axis=1,inplace=True)
x_train=x_train.to_numpy()
y_train=y_train.to_numpy()

#----------- TRAIN ANN CLASSIFYING 0,1,2 FROM 3 -----------#

#y_train_combined = y_train1.to_numpy()
#y_train_combined[y_train_combined == 1] = 0
#y_train_combined[y_train_combined == 2] = 0
#y_train_combined[y_train_combined == 3] = 1

#x_train, x_test_val, y_train, y_test_val = train_test_split(x_train, y_train, test_size=0.4, random_state=42)
model = ANN.MLP1(num_classes=4,epochs=420,predict=True)
model.train(x_train,np.squeeze(y_train))
#,x_test_val,np.squeeze(y_test_val)
model.predict_test_set(x_test.to_numpy())


#----------- TRAIN SVC CLASSIFYING 0,1,2 FROM 3 -----------#


#----------- TRAIN KNN CLASSIFYING 0,1,2 FROM 3 -----------#

#----------- COMBINE ALL CLASSIFIERS ---------#

#----------- MAKE SUBMISSION ---------#

dt = pd.DataFrame(data=model.prediction, columns=['y'])
dt['id'] = dt.index
dt = dt[['id', 'y']]
dt.to_csv('submission', header=True, index=False)