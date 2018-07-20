


#from skimage import io
#import numpy as np

#image = io.imread('00024.jpg')

#print(np.mean(image))
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import  os
from skimage import io
import cv2
from scipy.interpolate import interp1d
from scipy.stats import linregress

d=[]
x=np.array([60800,68000,75000,85800,95800,100500])
y=np.array([388,447.314,501.567,587.562,667.5,706.19])
h=[]
t=200./30.
u=np.array([6,13,23,32])

def best_fit_slope(x,y):
     m=((mean(x) * mean(y)) - mean(x*y))/ ((mean(x) * mean(x)) - mean(x*x))
     #b=mean(y)- m*mean(x)
     return m

slope= best_fit_slope(x,y)
print("slope:",slope)
for items in u:
    g =(t/200.)* items
    m=264.  * g   # Number of steps/sec =261.0
    k= slope  *  m
    h.append(k)  #Wavelength dispersion

x1=np.array([400,800,1600,3000])

f= interp1d(x,y, kind='linear')
l= interp1d(x1,h, kind='linear')
#print(linregress(x, y))
p= np.poly1d(np.polyfit(x, y,1))
plt.subplot(211)
plt.plot(x,y,'ro')
plt.plot(x,p(x),'b--')
#plt.plot(x,f(x),'b-')
plt.xlabel('Counter no.')
plt.ylabel('Wavelength(nm)')
plt.title('Wavelength - Counter no. relation ')
#slope, intercept = np.polyfit(np.log(y), np.log(x), 1)

plt.text(61000.,650.,'slope: 0.007982559357541521 nm/step',fontsize=10)
#plt.loglog(y, x, '--')
o= np.poly1d(np.polyfit(x1, h, 2))# polyfit of Wavelength v Slit width
plt.subplot(212)
plt.plot(x1,h,'yo', x1,o(x1),'r-')
plt.xlabel('Slit width (microns)')
plt.ylabel('fwhm (nm)')
plt.title('Resolution vs slit width (H-alpha)')
print(p(94000))
print('fwhm in nm:',o(100))



plt.show()
#print(x)
