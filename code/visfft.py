
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, print_function
import os
import pyedflib
import numpy as np
from math import sqrt,atan2
import scipy.fftpack as fftpack

def transform(signal,fT,plotName,save):
    from scipy.fftpack import fft

    yf = fft(signal)
    
    if save:
        import numpy as np
        import matplotlib.pyplot as plt
        
        N = len(signal)
        # sample spacing
        dT = fT / N
        
        xf = np.linspace(0.0, 0.5/dT, N/2)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
        ax1.grid()
        plt.savefig(plotName)
        plt.close(fig1)
    
    return yf
    
def main():
    from os.path import isfile, join
    from os import listdir
    
    pathData = "data/"
    if not os.path.exists(pathData):
        print('Path data input wrong ' + pathData)
        return

    plot=True
    pathPlots = "plots/"
    if not os.path.exists(pathPlots):
        os.makedirs(pathPlots)

    vDataFile = [pathData + f for f in listdir(pathData) if isfile(join(pathData, f)) and f[-4:]==".edf"]
    for dataFile in vDataFile:
        with pyedflib.EdfReader(dataFile) as f:
            pathPlotsFile = pathPlots + dataFile.split(".")[0].split("/")[-1] +"/"
            
            if not os.path.exists(pathPlotsFile ):
                os.makedirs(pathPlotsFile)
            
            #no trobo variable num canals en f
            for channel in range(1000):
                try:
                    signal = f.readSignal(channel)
                    duration = f.file_duration
                    plotName = pathPlotsFile + f.getLabel(channel) + ".png" 
                    transform(signal, duration, plotName,plot)
                except:
                    break
if __name__ == '__main__':
    main()