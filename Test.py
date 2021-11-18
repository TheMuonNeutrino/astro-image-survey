# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 11:37:05 2021

@author: hz3419
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from   scipy.stats import multivariate_normal

def gaussian2D(X,Y,a, mux, sigma_x, muy, sigma_y):
    #X=xy[0]
    #Y=xy[1]
    #a, mux, sigma_x, muy, sigma_y =(
            #params[0],params[1],params[2],params[3],params[4]
            #)
    X,Y = np.meshgrid(X,Y)
    
    gaus = a*np.exp(
            -(X-mux)**2/(2*sigma_x**2)-(Y-muy)**2/(2*sigma_y**2)
            )
    
        
    return gaus

xy1=np.array([np.linspace(-100,100,20),np.linspace(-100, 100, 20)])
#xy2=np.array([np.linspace(10,15,5),np.linspace(7, 15, 5)])
#params=np.array([100,0,60,0,60])
universe=gaussian2D(xy1[0],xy1[1],100,-60,20,-15,10)
universe+=gaussian2D(xy1[0],xy1[1],100,60,20,15,10)
plt.imshow(universe)
plt.show()
plt.colorbar()



#%%
'''
https://gist.github.com/
    gwgundersen/90dfa64ca29aa8c3833dbc6b03de44be#file-contour_2d_gaussian-py-L1

'''

