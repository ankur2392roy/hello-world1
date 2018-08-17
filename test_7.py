from scipy import signal
import pylab as plt
import numpy as np
from scipy.interpolate import splrep,splev


from matplotlib import pyplot as mp
from math import sqrt,pi
import csv
m,l= np.loadtxt('Hitran_o.txt').T
m=np.array(m)
l=np.array(l)

t=[]
b=[]
j=[]
fwhm=0.05
npts=3327
def gaussian(x,i, mu, fwhm):
    sig=fwhm/2.355
    a= 1./(sqrt(2.*pi)*sig)*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    b= i*a
    return b

u=np.linspace(0.05,4.0,396)
u=np.array(u)
u=u/2.355
x=np.linspace(541.83, 781.35, 3327)
f=np.zeros(x.shape[0])


while fwhm < 4.0:
     for c,d in zip(m,l):
        f=f+gaussian(x,d,c, fwhm)

     t.append(f)
     fwhm=fwhm+0.01

t=np.array(t)

for i in range(0,len(t)):
    b=t[i]
    b=np.array(b)
    y1= [(0.5*abs((y-b.max())/((b.max()-b.min()))) + 0.5) for y in b ]
    y1=np.array(y1)
    x1,p= np.loadtxt('Norm.txt').T
    p=np.array(p)
    y2= [(0.5*abs((y-p.min())/((p.max()-p.min()))) + 0.5) for y in p ]
    y2=np.array(y2)
    lags = np.arange(-npts + 1, npts)
    ccov = np.correlate(y1 - y1.mean(), y2 - y2.mean(), mode='full')
    ccor = ccov / (npts * y1.std() * y2.std())
    j.append(ccor.max()) 


#mp.plot()
j=np.array(j)
print(t[150])
#mp.plot(x,t[150])
mp.plot(u,j,'r')


#data = zip(x, f)
'''with open("Hitran_3.csv", "a") as k:
    coords = [map(str, tupl) for tupl in data]
    writer = csv.writer(k, delimiter=',')

    for line in coords:
        writer.writerow(line)'''

mp.show()
