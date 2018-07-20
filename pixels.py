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


image_dir = os.path.join(cwd,'images3')
files = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]

pixels=[]
x=[]
f=[]
y=[]

def gaus(x,a,x0,b):
    return a* np.exp(-(x-x0)**2/(2*(b**2)))

for image in files:
   img  = io.imread(os.path.join(image_dir,image))
   pixels.append(np.mean(img))

#df = pd.DataFrame(pixels)
#df.to_csv("Pixels_0.csv")
#plt.subplot(211)
#plt.plot(range(0,len(pixels)),pixels,'b-')
#plt.ylabel('PIXEL_Intensities')
#plt.title('3000microns')
#plt.axis([0.,200.,26.,36.])

#plt.subplot(212)
#a=mean(pixels)


x=range(0,len(pixels))
#=np.array([a for i in xrange(len(x))])
y=pixels
#print(pixels[73])
#f= interp1d(x,pixels, kind='linear')
#mean = sum(x*y)/sum(y)
#sigma =  np.sqrt(sum(y * (x - mean)**2) /sum(y))
#popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])
#f1=interp1d(range(0,len(pixels)),pixels, kind='cubic')
plt.plot(x,y,'r--')#,x,f(x),'g-')#range(0,len(pixels)),f1,'r-')
#plt.axhline(y=a, color='k', linestyle='-')
#plt.plot(x,gaus(x,*popt),'b+')
#plt.legend(['original','quadratic','cubic'])
plt.ylabel('PIXEL_Intensities')
plt.xlabel('Frame #')
plt.title('entry:42 microns, exit:74 microns')
#plt.axis([0.,74.,60.,70.])



plt.show()
