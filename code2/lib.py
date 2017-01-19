
def createFolder(nameFolder):
	import os
	if not os.path.isdir(nameFolder):
		try:
			os.makedirs(nameFolder)
		except:
			print("Imposible to create folder: " + str(nameFolder))
	else:
		print("Folder " + str(nameFolder) + " exist")


def filter(signal, finalTime, filSup, filInf):
	import numpy as np
	from scipy.fftpack import rfft, irfft, fftfreq

	W = fftfreq(signal.size, d=float(finalTime)/len(signal))
	f_signal = rfft(signal)

	cut_f_signal = f_signal.copy()
	cut_f_signal[(W<filInf)] = 0
	cut_f_signal[(W>filSup)] = 0

	cut_signal = irfft(cut_f_signal)

	return cut_signal

def getDicLabelChannel(fileEDF):
	dicLabelChannel={}
	for channel in range(64):
		dicLabelChannel[fileEDF.getLabel(channel)]=channel
	return dicLabelChannel

def getFileName(pacient,experiment):
	import params
	info = params.getParams()
	pathData = info["dataEDF"]
	folder = "S"+str(pacient).zfill(3)
	pathFile = pathData + "/" + folder
	fileName = folder + "R" + str(experiment).zfill(2) + ".edf"
	pathHostFile = pathFile + "/" + fileName
	return pathHostFile

def getFolder(pacient):
	import params
	info = params.getParams()
	pathData = info["dataEDF"]
	folder = "S"+str(pacient).zfill(3)
	return pathData+folder


def pearson_corr(Matrix):
    from scipy.stats import pearsonr
    return [[pearsonr(Matrix[x],Matrix[y])[0] for x in range(len(Matrix))] for y in range(len(Matrix))]    

