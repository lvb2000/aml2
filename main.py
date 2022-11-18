import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import feature_selection as fs

def get_data():
    x_train_ = pd.read_csv('X_train.csv').drop('id', axis=1)
    y_train_ = pd.read_csv('y_train.csv').drop('id', axis=1)
    x_test_ = pd.read_csv('X_test.csv').drop('id', axis=1)
    return x_train_, y_train_, x_test_

def make_submission(prediction_, name='submission.csv'):
    dt = pd.DataFrame(data=prediction_, columns=['y'])
    dt['id'] = dt.index
    dt = dt[['id', 'y']]
    dt.to_csv(name, header=True, index=False)

x_train_, y_train_, x_test_ = get_data()

signals = fs.biosppy(sampling_rate=1000)
x_train_ = signals.fit(x_train_,True)