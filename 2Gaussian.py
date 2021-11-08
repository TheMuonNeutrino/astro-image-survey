# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:37:08 2021

@author: hz3419
"""
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import astropy
from astropy.io import fits
from scipy import optimize
#%%
with fits.open('A1_mosaic.fits') as hdulist: # hdu as Header Data Unit
    data=hdulist[0].data

values,counts=np.unique(data.flatten(),return_counts=True)      
plt.scatter(values,counts,marker='o',s=5)
plt.xlabel('Pixel Values (Intensity)')
plt.ylabel('Pixel Counts (Number of pixels)')
plt.show()
#%%
mask= (values>3200)&(values<3800)&(counts<400000)
values=values[mask]
counts=counts[mask]

#plt.hist(data.flatten(), bins=10000)
def gaussian(x,a,mu,sigma,a2,mu2,sigma2):
    gaus1=a*sp.exp(-(x-mu)**2/(2*sigma**2))
    gaus2=a2*sp.exp(-(x-mu2)**2/(2*sigma2**2))
    return gaus1+gaus2

guess=[350000,3420,10,20000,3500,20]
po,po_cov=sp.optimize.curve_fit(gaussian,values,counts,guess)

print("The parameters are")
print(" Gaussian amplitude 1 = %.1f +/- %.1f" %(po[0],sp.sqrt(po_cov[0,0])))
print(" mu1 = %.1f +/- %.1f"%(po[1],sp.sqrt(po_cov[1,1])))
print(" sigma1 = %.1f +/- %.1f"%(po[2],sp.sqrt(po_cov[2,2])))
print(" Gaussian amplitude 2 = %.1f +/- %.1f" %(po[3],sp.sqrt(po_cov[3,3])))
print(" mu2 = %.1f +/- %.1f"%(po[4],sp.sqrt(po_cov[4,4])))
print(" sigma2 = %.1f +/- %.1f"%(po[5],sp.sqrt(po_cov[5,5])))


plt.plot(values,gaussian(values,po[0],po[1],po[2],po[3],po[4],po[5]),label='Fit results')
plt.scatter(values,counts, marker='x',label="Input Data",s=4,color='red')
plt.xlabel("Intensity")
plt.ylabel("Counts")
plt.legend()
plt.show()

#%%
