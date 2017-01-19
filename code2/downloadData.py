
def main():
	import os
	import urllib
	import lib
	import params

	info = params.getParams()
	pahtData=info["dataEDF"]
	webPage =info["webPage"]

	lib.createFolder(pahtData)
	
	onlineReader = urllib.URLopener()
	
	#http://www.physionet.org/pn4/eegmmidb/S001/S001R01.edf
	for volunter in range(1,110):

		folder = "S"+str(volunter).zfill(3)
		pathFile = pahtData + "/" + folder
		lib.createFolder(pathFile)

		for experiment in range(1,15):
			fileName = folder + "R" + str(experiment).zfill(2) + ".edf"
			pathOnlineFile = webPage + folder + "/" + fileName
			pathHostFile = pathFile + "/" + fileName

			print pathOnlineFile + " --> " + pathHostFile
			onlineReader.retrieve(pathOnlineFile,fileName)
			os.rename(fileName, pathHostFile)
			

if __name__ == '__main__':
    main()