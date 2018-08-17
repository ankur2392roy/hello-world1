from scipy import signal
import pylab as plt
import numpy as np
from scipy.interpolate import splrep,splev
import sys
import os
#sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
#sig_noise = sig + np.random.randn(len(sig))


sig=0.07/2.355
npts=3327
x,m= np.loadtxt('Hitran_3.txt').T
m=np.array(m)
y1= [(0.5*abs((y-m.max())/((m.max()-m.min()))) + 0.5) for y in m ]
y1=np.array(y1)
x1,l= np.loadtxt('Norm.txt').T
l=np.array(l)
y2= [(0.5*abs((y-l.min())/((l.max()-l.min()))) + 0.5) for y in l ]
y2=np.array(y2)
lags = np.arange(-npts + 1, npts)
ccov = np.correlate(y1 - y1.mean(), y2 - y2.mean(), mode='full')
ccor = ccov / (npts * y1.std() * y2.std())

fig, axs = plt.subplots(nrows=2)
fig.subplots_adjust(hspace=0.6)
ax = axs[0]
ax.plot(x, y1, 'b', label='y1')#y1: Template oxygen spectrum
ax.plot(x1, y2, 'r', label='y2')#y2: Solar spectrum
ax.set_ylabel('Normalized intensity')
ax.set_xlabel('Wavelength(nm)')
ax.set_title('entry:60microns, exit:134microns')
#ax.set_ylim(-10, 10)
ax.legend(loc='upper right', fontsize='small', ncol=2)

ax = axs[1]
ax.plot(lags, ccor)
#ax.set_ylim(-1.1, 1.1)
ax.set_ylabel('cross-correlation')
ax.set_xlabel('lag of y1 relative to y2')
ax.set_title(r'$\sigma = 0.030, max corr=0.5933 $')
maxlag = lags[np.argmax(ccor)]
print("max correlation is at lag %d" % maxlag)
print(sig)
print(ccor.max())



plt.show()

