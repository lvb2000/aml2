import pandas as pd
import numpy as np
import biosppy.signals.ecg as ecg

class biosppy:

    sampling_rate=1000

    def __init__(self,sampling_rate):
        self.sampling_rate=sampling_rate

    def fit(self,X,show=False):
        for i in X:
            signal = i.dropna()
            ecg.ecg(signal=signal, sampling_rate=self.sampling_rate, show=show)
        return X