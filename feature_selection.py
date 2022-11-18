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
        variances = np.empty([1,1000])
        count = 0
        for i in X:
            count+=1
            signal = X[i].dropna()

            # extract heartbeats with ecg function
            extracted_heartbeats = ecg.ecg(signal=signal, sampling_rate=self.sampling_rate, show=show)[4]

            #compute mean and variance of extracted heartbeats
            mean,variance = self.__mean_variance__(extracted_heartbeats)

            #add them into array
            mean=np.reshape(np.pad(mean, (0, 1000 - np.size(mean)), 'constant', constant_values=(np.nan, np.nan)),(1,1000))
            variance=np.reshape(np.pad(variance, (0, 1000 - np.size(variance)), 'constant', constant_values=(np.nan, np.nan)),(1,1000))
            means = np.concatenate((means,mean),axis=0)
            variances = np.concatenate((variances,variance),axis=0)

            if count==20:
                return means,variances
        return means,variances

    def plot(self,mean,variance):
        mean = mean[np.logical_not(np.isnan(mean))]
        variance = variance[np.logical_not(np.isnan(variance))]
        x=np.arange(0,np.size(mean)/self.sampling_rate,1/self.sampling_rate)
        plt.plot(x, mean, '-', color='gray')
        plt.fill_between(x, mean - variance, mean + variance,color='gray', alpha=0.2)
        plt.show()
        return
    def __mean_variance__(self,out):
        mean = np.mean(out,axis=0)
        variance = np.std(out,axis=0)
        return mean , variance

