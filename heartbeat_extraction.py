import pandas as pd
import numpy as np
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt

class biosppy:

    sampling_rate=300
    means = np.empty([1, 180])
    stds = np.empty([1, 180])

    def __init__(self,sampling_rate):
        self.sampling_rate=sampling_rate

    def fit(self,X,show=False):
        for i,row in X.iterrows():
            signal = row.dropna().to_numpy(dtype='float32')

            r_peaks = ecg.engzee_segmenter(signal, self.sampling_rate)['rpeaks']
            if len(r_peaks) >= 2:
                extracted_heartbeats = ecg.extract_heartbeats(signal, r_peaks, self.sampling_rate)['templates']
                #ecg.ecg(signal, self.sampling_rate,show=True)
                #compute mean and std of extracted heartbeats
                mean,std = self.__mean_variance__(extracted_heartbeats)

                #add them into array
                self.means = np.concatenate((self.means,np.reshape(mean,(1,180))),axis=0)
                self.stds = np.concatenate((self.stds,np.reshape(std,(1,180))),axis=0)


        self.means = np.delete(self.means, 0, 0)
        self.stds = np.delete(self.stds,0,0)
        return self.means,self.stds

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

    def toCSV(self):
        columns = []
        for i in range(self.means.shape[1]):
            columns = columns + ['x{i}'.format(i=i)]
        dt_mean = pd.DataFrame(data=self.means, columns=columns)
        dt_mean.to_csv('mean', header=True, index=False)
        dt_std = pd.DataFrame(data=self.stds, columns=columns)
        dt_std.to_csv('std', header=True, index=False)

