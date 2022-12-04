import pandas as pd
import numpy as np
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import pywt
from PyAstronomy import pyaC
import biosppy.signals.ecg as ecg

class feature_creation:
    # QRS
    R_index=[]
    RR_interval=np.empty(0)
    R_amplitude=np.empty(0)
    Q_positions = []
    Q_start_positions = []
    S_positions = []
    S_end_positions = []

    # P
    P_position = []
    P_amplitude = []
    P_std = []
    PR_interval = 0

    # mean of std
    mean_std = 0


    def createFeatures(self,std,mean):
        self.__meanStd__(std)
        self.__R__(mean)
        self.__QRS__(mean)


    def __meanStd__(self,std):
        self.mean_std = np.sum(std,axis=1)

    def __R__(self,mean):
        for i in mean:
            self.R_index += [np.argmax(i)]

    def __QRS__(self,mean):
        for i,m in enumerate(mean):
            df = pd.DataFrame(data=np.transpose(np.array([[m],[60]])), columns=["templates","rpeaks"])
            Q_position, Q_start_position = ecg.getQPositions(df)
            S_position, S_end_position = ecg.getSPositions(df)
            self.Q_positions.append(Q_position)
            self.Q_start_positions.append(Q_start_position)
            self.S_positions.append(S_position)
            self.S_end_positions.append(S_end_position)



    def getFeatures(self):
        features = pd.DataFrame(data=np.transpose(np.array([self.mean_std])), columns=['mean_std'])
        return features

    def plotMeanStd(self,y_train,means,variances):
        n=500
        y=np.ones(500)

        for i in range(10):
            mean = means.to_numpy()[i]
            variance = variances.to_numpy()[i]
            plt.figure()
            plt.plot(range(mean.shape[0]), mean, '-', color='gray')
            plt.fill_between(range(mean.shape[0]), mean - variance, mean + variance, color='gray', alpha=0.2)
            plt.vlines([self.R_index[i], self.P_position[i]], -1, 1)
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [mV]')
            plt.show()

        fig1, axs = plt.subplots(4, 1)
        fig1.tight_layout(pad=1.5)
        for i in range(4):
            index=(y_train.to_numpy() == i)[:, 0]
            x=self.mean_std[index]
            y=np.ones(5082)[index]
            #axs[i].scatter(x,y)
            axs[i].hist(x,20)
            axs[i].set_title('Class {i}'.format(i=i))
            axs[i].set_xlim(0, 15)

        plt.show()

    def find_time_center(data, show=False):
        tc = []
        for j, x in enumerate(data):
            xc, xi = pyaC.zerocross1d(np.array(range(len(x))), x, getIndices=True)
            tc.append(int(np.mean(xi)))

            if j % 1000 == 0 and show:
                plt.plot(range(len(x)), x)
                plt.vlines(tc[j], -0.25, 0.25)
                plt.show()

        return tc

def wavelet_decomp_recon(mean,show=False):
    mean_dwt=[]
    mode = pywt.Modes.smooth

    for j,x in enumerate(mean):
        w = pywt.Wavelet('sym4')
        a=x
        ca=[]
        cd=[]
        for i in range(5):
            (a,d) = pywt.dwt(a,w,mode)
            ca.append(a)
            cd.append(d)

        rec_a = []
        rec_d = []

        for i, coeff in enumerate(ca):
            coeff_list = [coeff, None] + [None] * i
            rec_a.append(pywt.waverec(coeff_list, w))

        mean_dwt.append(rec_a[3])

        for i, coeff in enumerate(cd):
            coeff_list = [None, coeff] + [None] * i
            rec_d.append(pywt.waverec(coeff_list, w))

        if j%1000==0 and show:

            fig = plt.figure()
            ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
            ax_main.set_title('Decomposition and Reconstruction of mean with discrete Wavelet transform')
            ax_main.plot(x)
            ax_main.set_xlim(0, len(x) - 1)

            for i, y in enumerate(rec_a):
                ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
                ax.plot(y, 'r')
                ax.set_xlim(0, len(y) - 1)
                ax.set_ylabel("A%d" % (i + 1))

            for i, y in enumerate(rec_d):
                ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
                ax.plot(y, 'g')
                ax.set_xlim(0, len(y) - 1)
                ax.set_ylabel("D%d" % (i + 1))

            plt.show()

    return np.array(mean_dwt)
