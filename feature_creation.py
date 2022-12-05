import pandas as pd
import numpy as np
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import pywt
from PyAstronomy import pyaC
import biosppy.signals.ecg as ecg
import math

class feature_creation:
    # QRS
    R_index=[]
    RR_interval=np.empty(0)
    R_amplitude=np.empty(0)
    Q_positions = []
    Q_start_positions = []
    S_positions = []
    S_end_positions = []
    tc = []

    # P
    P_position = []
    P_amplitude = []
    P_std = []
    PR_interval = 0

    # mean of std
    mean_std = 0

    # distance features
    height_per_index=1.5
    df1 = []
    df2 = []
    df3 = []
    df4 = []
    df5 = []
    df6 = []
    df7 = []
    df8 = []
    df_dwt1 = []
    df_dwt2 = []
    df_dwt3 = []
    df_dwt4 = []
    df_dwt5 = []
    df_dwt6 = []
    df_dwt7 = []
    df_dwt8 = []


    def createFeatures(self,std,mean):
        self.__meanStd__(std)
        self.__R__(mean)
        self.__QRS__(mean)
        self.find_time_center(mean, show=False)
        mean_dwt = wavelet_decomp_recon(mean)
        self.__distance_features__(mean,mean_dwt)


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

    def __distance_features__(self,mean,mean_dwt):
        for i,m in enumerate(mean):
           self.__distance_feature1__(i,m,False)
           self.__distance_feature2__(i,m,False)
           self.__distance_feature3__(i,m,False)
           self.__distance_feature4__(i,m,False)
           self.__distance_feature5__(i,m,False)
           self.__distance_feature6__(i,m,False)
           self.__distance_feature7__(i,m,False)
           self.__distance_feature8__(i,m,False)
        for i,m in enumerate(mean_dwt):
           self.__distance_feature1__(i, m,True)
           self.__distance_feature2__(i, m,True)
           self.__distance_feature3__(i, m,True)
           self.__distance_feature4__(i, m,True)
           self.__distance_feature5__(i, m,True)
           self.__distance_feature6__(i, m,True)
           self.__distance_feature7__(i, m,True)
           self.__distance_feature8__(i, m,True)

    def __distance_feature1__(self,i,m,dwt):
        dp=[]
        for j,v in enumerate(m):
            if j > self.Q_start_positions[i][0] and j<self.tc[i]:
                vmax=self.height_per_index*(self.tc[i]-j)/300
                if v<vmax and v>0:
                    dp.append(v)
        if len(dp)==0:
            if dwt:
                self.df_dwt1.append(0)
            else:
                self.df1.append(0)
        else:
            sum=0
            for j,p in enumerate(dp):
                if j+1 < len(dp):
                    sum+=math.sqrt(pow(abs(dp[j+1]-p),2)+pow((1/300),2))
            if dwt:
                self.df_dwt1.append(sum)
            else:
                self.df1.append(sum)


    def __distance_feature2__(self,i,m,dwt):
        dp=[]
        for j,v in enumerate(m):
            if j > self.Q_start_positions[i][0] and j<self.tc[i]:
                upper_bound=self.height_per_index*(self.tc[i]-self.Q_start_positions[i][0])
                vmax=self.height_per_index*(self.tc[i]-j)/300
                if v>vmax and v<upper_bound:
                    dp.append(v)
        if len(dp) == 0:
            if dwt:
                self.df_dwt2.append(0)
            else:
                self.df2.append(0)
        else:
            sum=0
            for j,p in enumerate(dp):
                if j+1 < len(dp):
                    sum+=math.sqrt(pow(abs(dp[j+1]-p),2)+pow((1/300),2))
            if dwt:
                self.df_dwt2.append(sum)
            else:
                self.df2.append(sum)

    def __distance_feature3__(self,i,m,dwt):
        dp=[]
        for j,v in enumerate(m):
            if j > self.Q_start_positions[i][0] and j<self.tc[i]:
                upper_bound=self.height_per_index*(self.tc[i]-self.Q_start_positions[i][0])
                vmax=self.height_per_index*(self.tc[i]-j)/300
                if v>-vmax and v<0:
                    dp.append(v)
        if len(dp) == 0:
            if dwt:
                self.df_dwt3.append(0)
            else:
                self.df3.append(0)
        else:
            sum=0
            for j,p in enumerate(dp):
                if j+1 < len(dp):
                    sum+=math.sqrt(pow(abs(dp[j+1]-p),2)+pow((1/300),2))
            if dwt:
                self.df_dwt3.append(sum)
            else:
                self.df3.append(sum)

    def __distance_feature4__(self,i,m,dwt):
        dp=[]
        for j,v in enumerate(m):
            if j > self.Q_start_positions[i][0] and j<self.tc[i]:
                upper_bound=self.height_per_index*(self.tc[i]-self.Q_start_positions[i][0])
                vmax=self.height_per_index*(self.tc[i]-j)/300
                if v>-upper_bound and v<-vmax:
                    dp.append(v)
        if len(dp) == 0:
            if dwt:
                self.df_dwt4.append(0)
            else:
                self.df4.append(0)
        else:
            sum=0
            for j,p in enumerate(dp):
                if j+1 < len(dp):
                    sum+=math.sqrt(pow(abs(dp[j+1]-p),2)+pow((1/300),2))
            if dwt:
                self.df_dwt4.append(sum)
            else:
                self.df4.append(sum)

    def __distance_feature5__(self,i,m,dwt):
        dp=[]
        for j,v in enumerate(m):
            if j < self.S_end_positions[i][0] and j>self.tc[i]:
                upper_bound=self.height_per_index*(self.tc[i]-self.Q_start_positions[i][0])
                vmax=self.height_per_index*(j-self.tc[i])/300
                if v>0 and v<vmax:
                    dp.append(v)
        if len(dp) == 0:
            if dwt:
                self.df_dwt5.append(0)
            else:
                self.df5.append(0)
        else:
            sum=0
            for j,p in enumerate(dp):
                if j+1 < len(dp):
                    sum+=math.sqrt(pow(abs(dp[j+1]-p),2)+pow((1/300),2))
            if dwt:
                self.df_dwt5.append(sum)
            else:
                self.df5.append(sum)

    def __distance_feature6__(self,i,m,dwt):
        dp=[]
        for j,v in enumerate(m):
            if j < self.S_end_positions[i][0] and j>self.tc[i]:
                upper_bound=self.height_per_index*(self.tc[i]-self.Q_start_positions[i][0])
                vmax=self.height_per_index*(j-self.tc[i])/300
                if v>vmax and v<upper_bound:
                    dp.append(v)
        if len(dp) == 0:
            if dwt:
                self.df_dwt6.append(0)
            else:
                self.df6.append(0)
        else:
            sum=0
            for j,p in enumerate(dp):
                if j+1 < len(dp):
                    sum+=math.sqrt(pow(abs(dp[j+1]-p),2)+pow((1/300),2))
            if dwt:
                self.df_dwt6.append(sum)
            else:
                self.df6.append(sum)

    def __distance_feature7__(self,i,m,dwt):
        dp=[]
        for j,v in enumerate(m):
            if j < self.S_end_positions[i][0] and j>self.tc[i]:
                upper_bound=self.height_per_index*(self.tc[i]-self.Q_start_positions[i][0])
                vmax=self.height_per_index*(j-self.tc[i])/300
                if v>-vmax and v<0:
                    dp.append(v)
        if len(dp) == 0:
            if dwt:
                self.df_dwt7.append(0)
            else:
                self.df7.append(0)
        else:
            sum=0
            for j,p in enumerate(dp):
                if j+1 < len(dp):
                    sum+=math.sqrt(pow(abs(dp[j+1]-p),2)+pow((1/300),2))
            if dwt:
                self.df_dwt7.append(sum)
            else:
                self.df7.append(sum)

    def __distance_feature8__(self,i,m,dwt):
        dp=[]
        for j,v in enumerate(m):
            if j < self.S_end_positions[i][0] and j>self.tc[i]:
                upper_bound=self.height_per_index*(self.tc[i]-self.Q_start_positions[i][0])
                vmax=self.height_per_index*(j-self.tc[i])/300
                if v>-upper_bound and v<-vmax:
                    dp.append(v)
        if len(dp) == 0:
            if dwt:
                self.df_dwt8.append(0)
            else:
                self.df8.append(0)
        else:
            sum=0
            for j,p in enumerate(dp):
                if j+1 < len(dp):
                    sum+=math.sqrt(pow(abs(dp[j+1]-p),2)+pow((1/300),2))
            if dwt:
                self.df_dwt8.append(sum)
            else:
                self.df8.append(sum)

    def getFeatures(self):
        features = pd.DataFrame(data=np.transpose(np.array([self.df1,self.df2,self.df3,self.df4,self.df5,self.df6,self.df7,self.df8,self.df_dwt1,self.df_dwt2,self.df_dwt3,self.df_dwt4,self.df_dwt5,self.df_dwt6,self.df_dwt7,self.df_dwt8])), columns=['df1','df2','df3','df4','df5','df6','df7','df8','df_dwt1','df_dwt2','df_dwt3','df_dwt4','df_dwt5','df_dwt6','df_dwt7','df_dwt8'])
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

    def find_time_center(self,data, show=False):
        for j, x in enumerate(data):
            xc, xi = pyaC.zerocross1d(np.array(range(len(x))), x, getIndices=True)
            self.tc.append(int(np.mean(xi)))

            if j % 1000 == 0 and show:
                plt.plot(range(len(x)), x)
                plt.vlines(self.tc[j], -0.25, 0.25)
                plt.show()


def wavelet_decomp_recon(mean,show=False):
    mean_dwt=[]
    mode = pywt.Modes.smooth

    for j,x in enumerate(mean):
        w = pywt.Wavelet('sym5')
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
