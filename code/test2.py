from __future__ import division, print_function, absolute_import, print_function
import os
import pyedflib
import numpy as np
from math import sqrt,atan2
import scipy.fftpack as fftpack

def transform_complex(list):
    
    reallist = []
    for i in list:
        reallist.append([sqrt(i.real*i.real+i.imag*i.imag),atan2(i.imag,i.real),])
    return reallist


if __name__ == '__main__':
    data_dir = os.path.join('.', 'data')
    test_data_file = os.path.join(data_dir, 'S001R01.edf')
    f = pyedflib.EdfReader(test_data_file)
    print("\nlibrary version: %s" % pyedflib.version.version)

    print("\ngeneral header:\n")

    print("file duration: %i seconds" % f.file_duration)
    print("startdate: %i-%i-%i" % (f.getStartdatetime().day,f.getStartdatetime().month,f.getStartdatetime().year))
    print("starttime: %i:%02i:%02i" % (f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second))
    print("patientcode: %s" % f.getPatientCode())
    print("gender: %s" % f.getGender())
    print("birthdate: %s" % f.getBirthdate())
    print("patient_name: %s" % f.getPatientName())
    print("patient_additional: %s" % f.getPatientAdditional())
    print("admincode: %s" % f.getAdmincode())
    print("technician: %s" % f.getTechnician())
    print("equipment: %s" % f.getEquipment())
    print("recording_additional: %s" % f.getRecordingAdditional())
    print("datarecord duration: %f seconds" % f.getFileDuration())
    print("number of datarecords in the file: %i" % f.datarecords_in_file)
    print("number of annotations in the file: %i" % f.annotations_in_file)

    channel = 3
    
    print("\nsignal parameters for the %d.channel:\n\n" % channel)

    print("label: %s" % f.getLabel(channel))
    print("samples in file: %i" % f.getNSamples()[channel])
    print("physical maximum: %f" % f.getPhysicalMaximum(channel))
    print("physical minimum: %f" % f.getPhysicalMinimum(channel))
    print("digital maximum: %i" % f.getDigitalMaximum(channel))
    print("digital minimum: %i" % f.getDigitalMinimum(channel))
    print("physical dimension: %s" % f.getPhysicalDimension(channel))
    print("prefilter: %s" % f.getPrefilter(channel))
    print("transducer: %s" % f.getTransducer(channel))
    print("samplefrequency: %f" % f.getSampleFrequency(channel))

    print(f.read_annotation())




'''
# Define signal.
Fs = f.getNSamples()[channel]  # Number samples
Tfin = f.getFileDuration() # Duration
Ts = Tfin/Fs
time = np.arange(0, Tfin, Ts)  # Time vector.









#x=f.readSignal(1)

#num = f.getNSamples()

x=f.readSignal(channel)
freq1 = np.fft.fft(x)
#rf = freq1.real
#imf = freq1.imag
#freq2 = (sqrt(rf*rf + imf*imf),atan2(imf,rf))
f1=open('signal.dat', 'w+')
f2=open('fourier.dat', 'w+')

freq2 = transform_complex(freq1)

#print(x.shape)


#for i in x:
#    print "%d %d" % (x[i],i)
    #f1.write("%s %s\n" % (int("0xFF" ,16), int("0xAA", 16)))
#f1=open('signal.dat', 'w+')
#f1.write(x)

#for i in x:
#    f1.write('{} {}'.format(x[i], i))

print ("Printing signal profile...")

count=0
for i in x:
    print(count,i)
    print(count,i,file=f1)
    #fprint(count,x[i])
    count += 0.00625

#print(freq,file=f2)    

### GOOD CODE
for i, x in enumerate(freq2):
    print(freq2[i][0],freq2[i][1],file=f2)

print("Frequencies", len(freq2))
#print("Frequencies", np.size(freq))

#print(f.getSignalLabels())

f1.close()
f2.close()'''