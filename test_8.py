from scipy import signal
import pylab as plt
import numpy as np
from scipy.interpolate import splrep,splev
import scipy.signal
import os
from matplotlib import pyplot as mp
from math import sqrt,pi
import csv


s=[]
m,l= np.loadtxt('Hyd_alpha.txt').T
m=np.array(m)
l=np.array(l)

g,h=np.loadtxt('NIST.txt').T
g=np.array(g)
h=np.array(h)

for i in m:
    i=i+0.337
    s.append(i)

s=np.array(s)


a=462.00

#data from test_1
b=781.35101
 #data from test_1
fwhm = (b-a)/601.


def gaussian(x,i, mu, fwhm):
    sig=fwhm/2.355
    a= 1./(sqrt(2.*pi)*sig)*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    b= i*a
    return b


x=np.linspace(600.155, 781.351, 342)
f=np.zeros(x.shape[0])

for c,d in zip(g,h):
    f=f+gaussian(x,d,c, fwhm)

f=np.array(f)
y1= [(0.5*abs((y-l.min())/((l.max()-l.min()))) + 0.5) for y in l ]
y1=np.array(y1)

y2= [(0.5*abs((y-f.min())/((f.max()-f.min()))) + 0.5) for y in f ]
y2=np.array(y2)

mp.plot(x,y2,'b', label='NIST')
mp.plot(s,y1,'r', label='DATA')
mp.legend(loc='upper right', fontsize='small', ncol=2)

recovered, remainder =signal.deconvolve(y2,y1)

recovered=np.array(recovered)
print(recovered)
mp.show()
