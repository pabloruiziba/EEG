# -*- coding: utf-8 -*-

def main():
	import numpy as np
	import matplotlib.pyplot as plt
	from pyedflib import EdfReader
	from os import listdir
	from os.path import isfile,join

	import lib
	import params

	info = params.getParams()

	pathPlots = info["plotsMS"]
	fM = info["Filter"]["upFilter"]
	fm = info["Filter"]["downFilter"]

	lib.createFolder(pathPlots)

	#vector on els items son diccionaris amb info del plot
	vplot = [{"plotname":"Fc","labels":["Fc5.","Fc3.","Fc1.","Fcz.","Fc2.","Fc4.","Fc6."]}]

	vDataFile = [lib.getFileName(pac,exp) for exp in info["experiment"] for pac in info["pacient"]]
	
	for dataFile in vDataFile:
	    with EdfReader(dataFile) as f:
	        dicLabelChannel= lib.getDicLabelChannel(f)

	        for plot in vplot:
	            
	            numSubPlots = len(plot["labels"])

	            fig, axarr = plt.subplots(numSubPlots, sharex=True)
	            relExtr = []
	            vDur = []
	            for subplot,label in enumerate(plot["labels"]):
	                vDur.append(f.file_duration)
	                y = lib.filter(f.readSignal(dicLabelChannel[label]), f.file_duration, fM,fm )
	                x=np.linspace(0,f.file_duration,len(f.readSignal(dicLabelChannel[label])))
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