
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, print_function
import os
import pyedflib
import numpy as np
from math import sqrt,pow
import scipy.fftpack as fftpack

################
import sys
import math
import random
import subprocess
##################

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

def filter(signal, fT, plotName,outputName,save):
    import numpy as np
    from scipy.fftpack import rfft, irfft, fftfreq

    time   = np.linspace(0,fT,len(signal))
    W = fftfreq(signal.size, d=fT/len(signal))
    f_signal = rfft(signal)

    # If our original signal time was in seconds, this is now in Hz    
    cut_f_signal = f_signal.copy()
    
    #pensar millor
    #filtre dinamic
    cut_f_signal[(W<0.1)] = 0
    cut_f_signal[(W>50)] = 0
    
    cut_signal = irfft(cut_f_signal)

    import pylab as plt
    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    #dibuixa el proces de filtratge
    if False:
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

    f1=open(outputName+'.dat', 'w+')

    for i in xrange(len(signal)):
        f1.write("{} {}\n".format(time[i], cut_signal[i]))

    f1.close()

    return cut_signal

    #ax2.plot(W,f_signal)
    #ax3.plot(W,cut_f_signal)
    #ax4.plot(time,cut_signal)
    #plt.savefig(plotName)
    #plt.close(f)

def pearson_corr(Matrix,num_points,num_channels):
    
    w, h = 64, 64 
    Corr = [[0 for x in range(w)] for y in range(h)]    

    for i in range (0,num_channels):
        av_x = np.mean(Matrix[i][:])
        for j in range (0,num_channels):
            sumXY=0
            sumX2=0
            sumY2=0
            av_y = np.mean(Matrix[j][:])
            
            for k in range (0,num_points):
                sumXY += (Matrix[i][k]-av_x)*(Matrix[j][k]-av_y)
                sumX2 += pow(Matrix[i][k]-av_x, 2)
                sumY2 += pow(Matrix[j][k]-av_y,2)

            Corr[i][j] = sumXY/(sqrt(sumX2)*sqrt(sumY2))

    return Corr

class Point:


    '''
    An point in n dimensional space
    '''
    def __init__(self, coords,name):
        '''
        coords - A list of values, one per dimension
        '''
        
        self.coords = coords
        self.n = len(coords)
        self.name = name
        
    def __str__(self):
        return str(self.name)
   
   
    def __repr__(self):
        return str(self.name)
   
    #def __repr__(self):
    #    return str(self.coords)

class Cluster:
    '''
    A set of points and their centroid
    '''
    
    def __init__(self, points):
       
        import sys
        import math
        import random
        import subprocess

        '''
        points - A list of point objects
        '''
        
        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        # The points that belong to this cluster
        self.points = points
        
        # The dimensionality of the points in this cluster
        self.n = points[0].n
        
        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")
            
        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()
        
    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)
    
    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        '''
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid) 
        return shift
    
    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
        
        return Point(centroid_coords,0)

def kmeans(points, k, cutoff):
    
    # Pick out k random points to use as our initial centroids
    initial = random.sample(points, k)
    
    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]
    
    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [ [] for c in clusters]
        clusterCount = len(clusters)
        
        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)
        
            # Set the cluster this point belongs to
            clusterIndex = 0
        
            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i+1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is, and set the point to belong
                # to that cluster
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            lists[clusterIndex].append(p)
        
        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0
        
        # As many times as there are clusters ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)
        
        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print("Converged after %s iterations" % loopCounter)
            break
    return clusters

def getDistance(a, b):
    '''
    Euclidean distance between two n-dimensional points.
    Note: This can be very slow and does not scale well
    '''
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points")
    
    ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(a.n),0.0)
    return math.sqrt(ret)

def makeRandomPoint(n, lower, upper):
    '''
    Returns a Point object with n dimensions and values between lower and
    upper in each of those dimensions
    '''
    p = Point([random.uniform(lower, upper) for i in range(n)],0)
    return p

def plotClusters(data):
    '''
    Use the plotly API to plot data from clusters.
    
    Gets a plot URL from plotly and then uses subprocess to 'open' that URL
    from the command line. This should open your default web browser.
    '''
    
    # List of symbols each cluster will be displayed using    
    symbols = ['circle', 'cross', 'triangle-up', 'square']

    # Convert data into plotly format.
    traceList = []
    for i, c in enumerate(data):
        data = []
        for p in c.points:
            data.append(p.coords)
        # Data
        trace = {}
        trace['x'], trace['y'] = zip(*data)
        trace['marker'] = {}
        trace['marker']['symbol'] = symbols[i]
        trace['name'] = "Cluster " + str(i)
        traceList.append(trace)
        # Centroid (A trace of length 1)
        centroid = {}
        centroid['x'] = [c.centroid.coords[0]]
        centroid['y'] = [c.centroid.coords[1]]
        centroid['marker'] = {}
        centroid['marker']['symbol'] = symbols[i]
        centroid['marker']['color'] = 'rgb(200,10,10)'
        centroid['name'] = "Centroid " + str(i)
        traceList.append(centroid)
    
    # Log in to plotly
    py = plotly(username=PLOTLY_USERNAME, key=PLOTLY_KEY)

    # Style the chart
    datastyle = {'mode':'markers',
             'type':'scatter',
             'marker':{'line':{'width':0},
                       'size':12,
                       'opacity':0.6,
                       'color':'rgb(74, 134, 232)'}}
    
    resp = py.plot(*traceList, style = datastyle)
    
    # Display that plot in a browser
    cmd = "open " + resp['url']
    subprocess.call(cmd, shell=True)

def main():
    from os.path import isfile, join
    from os import listdir
    
    pathData = "data/"
    if not os.path.exists(pathData):
        print('Path data input wrong ' + pathData)
        return

    plot=True
    pathPlots = "filter/"
    pathFiles = "output/"    
    if not os.path.exists(pathPlots):
        os.makedirs(pathPlots)

    vDataFile = [pathData + f for f in listdir(pathData) if isfile(join(pathData, f)) and f[-4:]==".edf"]
    
    for dataFile in vDataFile:
        with pyedflib.EdfReader(dataFile) as f:
            pathPlotsFile = pathPlots + dataFile.split(".")[0].split("/")[-1] +"/"
            pathOutputFile = pathFiles + dataFile.split(".")[0].split("/")[-1] +"/"
            
            if not os.path.exists(pathPlotsFile ):
                os.makedirs(pathPlotsFile)
            

            #list of time series
            w, h = 9760, 64 
            Matrix = [[0 for x in range(w)] for y in range(h)] 

            #no trobo variable num canals en f
            for channel in range(64):
            
                signal = f.readSignal(channel)
                duration = f.file_duration
                plotName = pathPlotsFile + f.getLabel(channel) + ".png"
                outputName = pathOutputFile + str(channel+1).zfill(2) + "_noise" 
                f_signal =filter(signal, duration, plotName, outputName, plot)
                for i in range(9760):
                    Matrix[channel][i]=f_signal[i]

            Corr = pearson_corr(Matrix,9760,64)
            print(Corr[63][63])

            # How many points are in our dataset?
            num_points = 10
            
            # For each of those points how many dimensions do they have?
            dimensions = 2
            
            # Bounds for the values of those points in each dimension
            lower = 0
            upper = 200
            
            # The K in k-means. How many clusters do we assume exist?
            num_clusters = 6
            
            # When do we say the optimization has 'converged' and stop updating clusters
            
            opt_cutoff = 0.0001
            
            # Generate some points
#            points = [makeRandomPoint(dimensions, lower, upper) for i in xrange(num_points)]

            points = []

            for i in range (0,64):
                points.append(Point(Corr[i][:],i+1))


#            print(points)
            
            # Cluster those data!
            clusters = kmeans(points, num_clusters, opt_cutoff)

            # Print our clusters
            for i,c in enumerate(clusters):
                for p in c.points:
                    print ("Cluster: ", i, "\t Point :", p)
                
if __name__ == '__main__':
    main()
