import pandas as pd
import numpy as np
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt

class feature_creation:
    # QRS
    R_index=[]
    RR_interval=np.empty(0)
    R_amplitude=np.empty(0)
    Q_ampltiude=np.empty(0)
    QRS_duration=np.empty(0)

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
        self.__P__(std,mean)

    def __meanStd__(self,std):
        self.mean_std = np.sum(std,axis=1)

    def __R__(self,mean):
        for i in mean:
            self.R_index += [np.argmax(i)]
    def __P__(self,std,mean):
        template_p_position_max = (40)
        for i in range(len(mean)):
            template_left = mean[i,0: template_p_position_max + 1]
            P_position= np.argmax(template_left)
            self.P_position += [P_position]
            self.P_amplitude += [mean[i,P_position]]
            self.P_std += [std[i,P_position]]
        self.PR_interval=np.subtract(np.array(self.R_index),np.array(self.P_position))


    def getFeatures(self):
        features = pd.DataFrame(data=np.transpose(np.array([self.mean_std,self.P_amplitude,self.P_std,self.PR_interval])), columns=['mean_std','P_amplitude','P_std','PR_interval'])
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