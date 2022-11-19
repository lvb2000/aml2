## imports
import numpy as np
import pandas as pd
import feature_selection as fs
import matplotlib.pyplot as plt

## import data
def get_data():
    x_train_ = pd.read_csv('X_train.csv').drop('id', axis=1)
    y_train_ = pd.read_csv('y_train.csv').drop('id', axis=1)
    x_test_ = pd.read_csv('X_test.csv').drop('id', axis=1)
    return x_train_, y_train_, x_test_

x_train_, y_train_, x_test_ = get_data()

## Plot sample signal for each class

def plot_sample_signals_for_each_class(data,labels):
    labels_array = labels['y'].to_numpy()
    num_classes = 4
    sample_signal_ids = []
    for class_id in range(num_classes):
        sample_signal_ids.append(int(np.argwhere(labels_array == class_id)[0]))

    # Some matplotlib setting
    plt.rcParams["figure.figsize"] = (30, 20)
    plt.rcParams['lines.linewidth'] = 5
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 32
    plt.rcParams['axes.labelsize'] = 48
    plt.rcParams['axes.titlesize'] = 48

    fig, axs = plt.subplots(4, 1)

    seconds = np.arange(0, 600) / 30
    x_labels = [0, 5, 10, 15, 20]

    for class_id in range(num_classes):
        ax = axs[class_id]
        # ax.set_title("Class {}".format(class_id))

        measurements = data.loc[sample_signal_ids[class_id]].dropna().to_numpy(dtype='float32')
        # Get a subsequence of a signal and downsample it for visualization purposes
        measurements = measurements[1000:7000:10]
        # convert volts to millivolts
        measurements /= 1000
        ax.plot(seconds, measurements, color='k')
        ax.set_xticks(x_labels)

    # Display x- and y-labels for the whole plot
    ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mV]')
    plt.show()

plot_sample_signals_for_each_class(x_train_,y_train_)

## extract features

signals = fs.biosppy(sampling_rate=300)
means, stds = signals.fit(x_train_,show=False)
signals.plot(means[0,:], stds[0, :])
signals.plot(means[1,:], stds[1, :])
signals.plot(means[2,:], stds[2, :])
signals.plot(means[3,:], stds[3, :])

print(y_train_[0:4])
