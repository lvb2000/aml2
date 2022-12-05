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
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

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
"""
features.plotMeanStd(y_train1,mean_train,std_train)
"""
#mean_dwt = fc.wavelet_decomp_recon(mean_train.to_numpy(),True)
#tc = fc.find_time_center(mean_train.to_numpy(),True)

#----------- IMPORT FEATURES -----------#
"""
x_train = pd.read_csv('features_ata.csv')
x_test = pd.read_csv('test_features_ata.csv')
y_train = pd.read_csv('y_train.csv')
x_train.drop(x_train.columns[0],axis=1,inplace=True)
x_test.drop(x_test.columns[0],axis=1,inplace=True)
y_train.drop(y_train.columns[0],axis=1,inplace=True)
x_train=x_train.to_numpy()
y_train=y_train.to_numpy()
"""
x_train=df.to_numpy()
y_train=y_train1.to_numpy()

#----------- TRAIN ANN1 -----------#

x_train, x_test_val, y_train, y_test_val = train_test_split(x_train, y_train, test_size=0.4, random_state=42)
model1 = ANN.MLP(num_classes=4,epochs=500,predict=False,MLP='MLP1')
model1.train(x_train,np.squeeze(y_train),x_test_val,np.squeeze(y_test_val))
#,x_test_val,np.squeeze(y_test_val)
#model1.predict_test_set(x_test.to_numpy())

#----------- TRAIN ANN2 -----------#
"""
model3 = ANN.MLP(num_classes=4,epochs=1300,predict=True,MLP='MLP3')
model3.train(x_train,np.squeeze(y_train))
#,x_test_val,np.squeeze(y_test_val)
model3.predict_test_set(x_test.to_numpy())

#----------- NORMALIZE DATA -----------#

scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.fit_transform(x_test)

#----------- BALANCE DATA -----------#

#smote = SMOTE(random_state = 14)
#x_train_balanced, y_train_balanced = smote.fit_resample(x_train_std, y_train)

#----------- TRAIN SVC -----------#

#svc = SVC(class_weight='balanced')
#svc.fit(x_train_balanced, y_train_balanced)
#svc_prediction = svc.predict(x_test_std)

#----------- TRAIN KNN -----------#
# The best k has been found to be:
k=16
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(x_train_std, np.squeeze(y_train))
knn_prediction = neigh.predict(x_test_std)

#----------- COMBINE ALL CLASSIFIERS ---------#

#----------- MAKE SUBMISSION ---------#

predictions = np.array([knn_prediction,model1.prediction,model3.prediction])
prediction = np.median(predictions,axis=0)
prediction = np.round(prediction,0)

dt = pd.DataFrame(data=prediction, columns=['y'])
dt['id'] = dt.index
dt = dt[['id', 'y']]
dt.to_csv('submission', header=True, index=False)
"""