# imports
import numpy as np
import pandas as pd
import heartbeat_extraction as he
import matplotlib.pyplot as plt

#----------- IMPORT RAW ECG SIGNALS -----------#
"""
x_train_ = pd.read_csv('X_train.csv').drop('id', axis=1)
"""

#----------- EXTRACT HEARTBEAT SIGNALS -----------#
"""
signals = he.biosppy(sampling_rate=300)
signals.fit(x_train_,show=False)
signals.toCSV()
"""

#----------- IMPORT EXTRACTED HEARTBEATS FROM CSV -----------#

mean_train = pd.read_csv('mean')
std_train = pd.read_csv('std')
y_train = pd.read_csv('y_train.csv').drop('id', axis=1)

