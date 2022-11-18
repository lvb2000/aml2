import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import feature_selection as fs


def get_data():
    x_train_ = pd.read_csv('X_train.csv').drop('id', axis=1)
    y_train_ = pd.read_csv('y_train.csv').drop('id', axis=1)
    x_test_ = pd.read_csv('X_test.csv').drop('id', axis=1)
    return x_train_, y_train_, x_test_

x_train_, y_train_, x_test_ = get_data()

def make_submission(prediction_, name='submission.csv'):
    dt = pd.DataFrame(data=prediction_, columns=['y'])
    dt['id'] = dt.index
    dt = dt[['id', 'y']]
    dt.to_csv(name, header=True, index=False)

signals = fs.biosppy(sampling_rate=1000)
means,variances = signals.fit(x_train_)
signals.plot(means[1,:],variances[1,:])
signals.plot(means[2,:],variances[2,:])
signals.plot(means[3,:],variances[3,:])
signals.plot(means[4,:],variances[4,:])


