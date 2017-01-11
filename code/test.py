import pyedflib
import numpy as np
f = pyedflib.EdfReader("data/S001R01.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
	sigbufs[i, :] = f.readSignal(i)
