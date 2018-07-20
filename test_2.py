'''import numpy as np
import matplotlib.pyplot as plt
import  os
from skimage import io
import cv2
from scipy.interpolate import interp1d
from scipy.stats import linregress
from statistics import mean
from matplotlib import style

style.use('fivethirtyeight')

x=np.array([60800,68000,75000,85800,95800,100500])
y=np.array([388,447.14,501.5,587.4,667.51,706.19])


def best_fit_slope(x,y):
     m=((mean(x) * mean(y)) - mean(x*y))/ ((mean(x) * mean(x)) - mean(x*x))
     b=mean(y)- m*mean(x)
     return m,b

a,b = best_fit_slope(x,y)

print (a,b)
l=(a*95800 + b)
u=[(a*i)+b for i in x]
print(l)
plt.plot(x,y,'ro')
plt.plot(x,u)
plt.show()'''

import numpy as np
import matplotlib.pyplot as plt
import  os
from skimage import io
import cv2
from scipy.interpolate import interp1d
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy import array
from statistics import mean

cwd = os.getcwd()


image_dir = os.path.join(cwd,'images4')
files = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]

pixels=[]
x=[]
f=[]
y=[]

#def gaus(x,a,x0,b):
    #return a* np.exp(-(x-x0)**2/(2*(b**2)))

for image in files:
   img  = io.imread(os.path.join(image_dir,image))
   pixels.append(np.mean(img))



a=302.3252557833629 #data from test_1
b=861.188637015141 #data from test_1
c = (b-a)/1056.
for i in range(0,1057,1):
    g=a + (c*i)
    f.append(g)


x=range(0,len(pixels))
y=pixels
#print(pixels[73])
plt.subplot(211)
plt.plot(x,y,'r--')

plt.ylabel('PIXEL_Intensities')
plt.xlabel('Frame #')
plt.title('entry:40 microns, exit:60 microns')#exposure 1/4 sec

plt.subplot(212)
plt.plot(f,y,'g--')
plt.ylabel('PIXEL Intensities')
plt.xlabel('Wavelength(nm)')
plt.title('entry:40 microns, exit:60 microns')

#print(iloc(min(pixels))


plt.show()
