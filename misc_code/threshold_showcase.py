# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:55:54 2021

@author: hz3419
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

randata=np.loadtxt('DemoData.txt',delimiter=',')
x1=randata[0]
y1=randata[1]
threshold=np.array([np.zeros(len(y1)),np.zeros(len(y1)),np.zeros(len(y1))])

for i in range(0,len(x1)):
    threshold[0][i]=23
    threshold[1][i]=14
    threshold[2][i]=7

demodata=np.array([x1,y1])
np.savetxt('DemonData.txt',demodata,delimiter=',')

#plt.plot(x1,threshold[0],c='blue')
plt.plot(x1,threshold[1],c='blue')
plt.plot(x1,threshold[2],c='blue')
plt.bar(x1,y1,width=1.3,color='orange')
plt.xlabel('1D Pixel Position')
plt.ylabel('Pixel Intensity')
plt.title('Object Identification 1D')
