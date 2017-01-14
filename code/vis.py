# -*- coding: utf-8 -*-

def filter(signal, fT):
    import numpy as np
    from scipy.fftpack import rfft, irfft, fftfreq
    
    W = fftfreq(signal.size, d=float(fT)/len(signal))
    f_signal = rfft(signal)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W<11)] = 0
    cut_f_signal[(W>12)] = 0
    
    return [np.linspace(0,fT,len(signal)),irfft(cut_f_signal)]

def main():
    from os.path import isfile, join, exists 
    from os import listdir, makedirs
    import numpy as np
    import matplotlib.pyplot as plt
    
    #!!!!!!!!!!!!!!!!!!!!!!!!
    test = True
    #!!!!!!!!!!!!!!!!!!!!!!!!
    
    pathData = "data/"
    if not exists(pathData):
        print('Path data input wrong ' + pathData)
        return

    pathPlots = "plots_vis/"
    if not exists(pathPlots):
        makedirs(pathPlots)

    
    #all labels with the exacly name
    #"Fc5.","Fc3.","Fc1.","Fcz.","Fc2.","Fc4.","Fc6.",
    #"C5..","C3..","C1..","Cz..","C2..","C4..","C6..",
    #"Cp5.","Cp3.","Cp1.","Cpz.","Cp2.","Cp4.","Cp6.",
    #"Fp1.","Fpz.","Fp2.",
    #"Af7.","Af3.","Afz.","Af4.","Af8.",
    #"F7..","F5..","F3..","F1..","Fz..","F2..","F4..","F6..","F8..",
    #"Ft7.","Ft8.",
    #"T7..","T8..","T9..","T10.",
    #"Tp7.","Tp8.",
    #"P7..","P5..","P3..","P1..",
    #"Pz..",
    #"P2..","P4..","P6..","P8..",
    #"Po7.","Po3.","Poz.","Po4.","Po8.",
    #"O1..","Oz..","O2..",
    #"Iz.."
    
    #vector on els items son diccionaris amb info del plot
    vplot = [{"plotname":"Fc","labels":["Fc5.","Fc3.","Fc1.","Fcz.","Fc2.","Fc4.","Fc6."]}]
    
    vDataFile = [pathData + f for f in listdir(pathData) if isfile(join(pathData, f)) and f[-4:]==".edf"]
    
    if test:
        vDataFile = [vDataFile[0]]
        vplot = [vplot[0]]


    from pyedflib import EdfReader
    for dataFile in vDataFile:
        with EdfReader(dataFile) as f:
            #el dic no tindria que ser el mateix per tots els fitxers
            dicLabelChannel={}

            #no trobo variable num canals en f
            for channel in range(64):
                dicLabelChannel[f.getLabel(channel)]=channel

            for plot in vplot:
                
                numSubPlots = len(plot["labels"])

                fig, axarr = plt.subplots(numSubPlots, sharex=True)
                relExtr = []
                vDur = []
                for subplot,label in enumerate(plot["labels"]):
                    vDur.append(f.file_duration)
                    x,y = filter(f.readSignal(dicLabelChannel[label]), f.file_duration)
                    relExtr.append(max(y))
                    relExtr.append(min(y))
                    axarr[subplot].plot(x, y)
                    
                    title = axarr[subplot].set_title(label)
                    offset = np.array([1.05, 0.0])
                    ax0label = axarr[subplot].set_ylabel('')
                    title.set_position(ax0label.get_position() + offset)
                    title.set_rotation(-90)


                axis = [0,max(vDur),min(relExtr),max(relExtr)]
                for subplot in range(len(plot["labels"])):
                    axarr[subplot].axis(axis)
                
                plt.savefig(pathPlots+plot["plotname"]+".png")

if __name__ == '__main__':
    main()