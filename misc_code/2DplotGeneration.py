# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:14:35 2021

@author: hz3419
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import random
import copy

rng = random.default_rng(456)


y=np.zeros(41)
threshold=np.array([np.zeros(len(y)),np.zeros(len(y)),np.zeros(len(y))])
x=np.linspace(0,len(y),len(y))

def gaus (x,y):
   param1=np.array([20,10,len(y)/10])
   param2=np.array([7,25,len(y)/10])
   param3=np.array([5,30,len(y)/10])
   
   gaus1=param1[0]*sp.exp(-(x-param1[1])**2/(2*param1[2]**2))
   gaus2=param2[0]*sp.exp(-(x-param2[1])**2/(2*param2[2]**2))
   gaus3=param3[0]*sp.exp(-(x-param3[1])**2/(2*param3[2]**2))
   
   return gaus1+gaus2+gaus3    
    
for i in range(0,len(x)-1):
    y[i]=rng.integers(0,4)
    y[i]+=gaus(x[i],y)
        
#%%
x1=copy.deepcopy(x)
y1=copy.deepcopy(y)
for i in range(0,len(x)):
    threshold[0][i]=23
    threshold[1][i]=5
    threshold[2][i]=7

demodata=np.array([x1,y1])
np.savetxt('DemoData.txt',demodata,delimiter=',')

#plt.plot(x1,threshold[0],c='blue')
plt.plot(x1,threshold[1],c='blue')
plt.plot(x1,threshold[2],c='blue')
plt.bar(x1,y1,width=1.2,color='orange')
plt.xlabel('1D Pixel Position')
plt.ylabel('Pixel Intensity')
plt.title('Object Identification 1D')
plt.show()
