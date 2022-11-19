import pandas as pd
import numpy as np
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt

class biosppy:

    sampling_rate=300

    def __init__(self,sampling_rate):
        self.sampling_rate=sampling_rate

    def fit(self,X,show=False):
        means= np.empty([1,1000])
        stds = np.empty([1,1000])
        for i,row in X.iterrows():
            signal = row.dropna().to_numpy(dtype='float32')

            r_peaks = ecg.engzee_segmenter(signal, self.sampling_rate)['rpeaks']
            if len(r_peaks) >= 2:
                extracted_heartbeats = ecg.extract_heartbeats(signal, r_peaks, self.sampling_rate)['templates']

                #compute mean and std of extracted heartbeats
                mean,std = self.__mean_variance__(extracted_heartbeats)

                #add them into array
                mean=np.reshape(np.pad(mean, (0, 1000 - np.size(mean)), 'constant', constant_values=(np.nan, np.nan)),(1,1000))
                std=np.reshape(np.pad(std, (0, 1000 - np.size(std)), 'constant', constant_values=(np.nan, np.nan)),(1,1000))
                means = np.concatenate((means,mean),axis=0)
                stds = np.concatenate((stds,std),axis=0)


        means = np.delete(means, 0, 0)
        stds = np.delete(stds,0,0)
        return means,stds

    def plot(self,mean,variance):
        mean = mean[np.logical_not(np.isnan(mean))]
        variance = variance[np.logical_not(np.isnan(variance))]
        plt.figure()
        plt.plot(range(mean.shape[0]), mean, '-', color='gray')
        plt.fill_between(range(mean.shape[0]), mean - variance, mean + variance,color='gray', alpha=0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [mV]')
        plt.show()
        return
    def __mean_variance__(self,out):
        mean = np.mean(out,axis=0)
        std = np.std(out,axis=0)
        return mean , std

