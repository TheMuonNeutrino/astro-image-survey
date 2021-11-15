# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:55:54 2021

@author: hz3419
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import random
import copy

seed = 456 + 5
rng = random.default_rng(seed)

def gaus (x,y):
   param1=np.array([20,10,len(y)/10])
   param2=np.array([14,25,len(y)/10])
   #param3=np.array([5,30,len(y)/10])
   
   gaus1=param1[0]*np.exp(-(x-param1[1])**2/(2*param1[2]**2))
   gaus2=param2[0]*np.exp(-(x-param2[1])**2/(2*param2[2]**2))
   #gaus3=param3[0]*sp.exp(-(x-param3[1])**2/(2*param3[2]**2))
   
   return gaus1+gaus2#+gaus3

y=np.zeros(41)
threshold=np.array([np.zeros(len(y)),np.zeros(len(y)),np.zeros(len(y))])
x=np.linspace(-2,len(y),len(y))    
    
for i in range(0,len(x)-1):
    y[i]+=gaus(x[i],y)
    y[i]+=rng.integers(0,y[i]//4+1)
        
x1=copy.deepcopy(x) + 2
y1=copy.deepcopy(y)
for i in range(0,len(x)):
    threshold[0][i]=23
    threshold[1][i]=9
    threshold[2][i]=6.5

demodata=np.array([x1,y1])

fig, ax = plt.subplots(1,1)

#plt.plot(x1,threshold[0],c='C1')
plt.plot(x1,threshold[1],c='C0')
plt.plot(x1,threshold[2],c='C0')
cuts = [6,17,19,20,21,30]
colorNoRegion = '#f8c471'
regions = [
    (colorNoRegion, slice(0,cuts[0])),
    ('#abebc6', slice(cuts[0],cuts[1])),
    (colorNoRegion, slice(cuts[1],cuts[2])),
    (colorNoRegion, slice(cuts[3],cuts[4])),
    ('#f1948a', slice(cuts[4],cuts[5])),
    (colorNoRegion, slice(cuts[5],None)),
    ('#bb8fce',slice(cuts[2],cuts[3])),
]
bars = []
for color, index in regions:
    bars.append(plt.bar(x1[index],y1[index],width=1.2,color=color))

bars[1].set_label('Region A')
bars[4].set_label('Region B')
bars[6].set_label('Contested Minima')
#plt.bar(x1,y1)
plt.xlabel('Pixel Position')
plt.ylabel('Pixel Intensity [arb.]')
plt.tick_params(
    axis='both',
    which='both', 
    #bottom=False, 
    top=False,
    left=False,
    right=False, 
    #labelbottom=False,
    labelleft=False) 
#plt.title('Object Identification 1D')
plt.legend()
plt.show()
