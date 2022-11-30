import pandas as pd
import numpy as np
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt

class feature_creation:
    RR_interval=np.empty(0)
    R_amplitude=np.empty(0)
    Q_ampltiude=np.empty(0)
    QRS_duration=np.empty(0)
    mean_std=0

    def createFeatures(self,std,mean):
        self.__meanStd__(std,mean)

    def __meanStd__(self,std,mean):
        self.mean_std = np.mean(std,axis=1)

    def getFeatures(self):
        features = pd.DataFrame(data=self.mean_std, columns=['mean_std'])
        return features

    def plotMeanStd(self,y_train):
        n=500
        y=np.ones(500)

        plt.figure(1)
        plt.scatter(self.mean_std[0:500],y,c=y_train.to_numpy()[0:500])
        plt.legend([0, 1, 2, 3])
        plt.show()

        fig1, axs = plt.subplots(4, 1)
        fig1.tight_layout(pad=1.5)
        for i in range(4):
            ms = self.mean_std.to_numpy()
            index=(y_train.to_numpy() == i)[:, 0]
            x=ms[index]
            y=np.ones(5082)[index]
            axs[i].scatter(x,y)
            axs[i].set_title('Class {i}'.format(i=i))
        plt.show()