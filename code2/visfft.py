# -*- coding: utf-8 -*-

def main():
    import lib
    from params import getParams, getCoords
    import pyedflib
    import kmeans

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.misc import imread
    import matplotlib.cbook as cbook

    info =getParams()

    lM=info["Filter"]["upFilter"]
    lm=info["Filter"]["downFilter"]

    vDataFile = [lib.getFileName(pac,exp) for exp in info["experiment"] for pac in info["pacient"]]
    
    numChanels = 64
    
    for dataFile in vDataFile:
    
        with pyedflib.EdfReader(dataFile) as f:

            #list of time series
            Matrix=[lib.filter(f.readSignal(channel), f.file_duration, lM,lm) for channel in range(numChanels)]
            Corr = lib.pearson_corr(Matrix)

            num_clusters = 6
            opt_cutoff = 0.0001
            
            points = [kmeans.Point(Corr[i],i+1) for i in range(numChanels)]
            
            # Cluster those data!
            clusters = kmeans.kmeans(points, num_clusters, opt_cutoff)

        colors=[0]*numChanels

        for i,c in enumerate(clusters):
            for p in c.points:
                colors[p.name-1] =i
                #print(str(p.name) + " --> " + str(i))
        
        img = imread("plots/64_channel_sharbrough.png")

        dicC =getCoords()

        x=[dicC[key][0] for key in dicC]
        y=[300+dicC[key][1]+90 for key in dicC]
        
        plt.scatter(x,y,zorder=1,s=[150]*numChanels, c=colors)
        plt.imshow(img, zorder=0, extent=[18.0, 282.0, 80.0, 300.0])
        plt.show()


if __name__ == '__main__':
    main()
