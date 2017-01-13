
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

def filter(signal, fT, plotName,save):
    import numpy as np
    from scipy.fftpack import rfft, irfft, fftfreq

    time   = np.linspace(0,fT,len(signal))
    W = fftfreq(signal.size, d=fT/len(signal))
    f_signal = rfft(signal)

    # If our original signal time was in seconds, this is now in Hz    
    cut_f_signal = f_signal.copy()
    
    #pensar millor
    #filtre dinamic
    cut_f_signal[(W>12)] = 0
    cut_f_signal[(W<11)] = 0
    
    cut_signal = irfft(cut_f_signal)

    import pylab as plt
    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    #dibuixa el proces de filtratge
    if True:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(time,signal)
        ax1.plot(time,cut_signal)
        plt.savefig(plotName.split(".")[0]+"_1"+plotName.split(".")[1])
        plt.close(fig1)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(W,f_signal)
        plt.savefig(plotName.split(".")[0]+"_2"+plotName.split(".")[1])
        plt.close(fig1)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(W,cut_f_signal)
        plt.savefig(plotName.split(".")[0]+"_3"+plotName.split(".")[1])
        plt.close(fig1)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(time,cut_signal)
    plt.savefig(plotName.split(".")[0]+"_4"+plotName.split(".")[1])
    plt.close(fig1)

    #ax2.plot(W,f_signal)
    #ax3.plot(W,cut_f_signal)
    #ax4.plot(time,cut_signal)
    #plt.savefig(plotName)
    #plt.close(f)
    
def main():
    from os.path import isfile, join
    from os import listdir
    
    pathData = "data/"
    if not os.path.exists(pathData):
        print('Path data input wrong ' + pathData)
        return

    plot=True
    pathPlots = "filter/"
    if not os.path.exists(pathPlots):
        os.makedirs(pathPlots)

    vDataFile = [pathData + f for f in listdir(pathData) if isfile(join(pathData, f)) and f[-4:]==".edf"]
    
    for dataFile in vDataFile:
        with pyedflib.EdfReader(dataFile) as f:
            pathPlotsFile = pathPlots + dataFile.split(".")[0].split("/")[-1] +"/"
            
            if not os.path.exists(pathPlotsFile ):
                os.makedirs(pathPlotsFile)
            
            #no trobo variable num canals en f
            for channel in range(64):
            
                signal = f.readSignal(channel)
                duration = f.file_duration
                plotName = pathPlotsFile + f.getLabel(channel) + ".png" 
                filter(signal, duration, plotName,plot)
                
if __name__ == '__main__':
    main()