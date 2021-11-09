import numpy as np
import matplotlib.pyplot as plt
import functools
import warnings
from os import path

from .utilities import pad

def gaussian(x,u,sigma,A):
    return A / (sigma*np.sqrt(2*np.pi)) * np.exp( -(x-u)**2 /(2*sigma**2) )

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, cmax, rmin, rmax

def indexFromBbox(bbox):
    cmin, cmax, rmin, rmax = bbox
    return (
        slice(rmin,rmax+1),
        slice(cmin,cmax+1)
    )

@functools.cache
def circularMask(radius):
    y,x = np.ogrid[-radius: radius+1, -radius: radius+1]
    mask = x**2+y**2 <= radius**2
    mask = mask.astype(int)
    return mask

def squareMaskAroundPoint(point,r,layerShape):
        rmin = np.max([point[0] - r,0])
        rmax = np.min([point[0] + r,layerShape[0]-1])
        cmin = np.max([point[1] - r,0])
        cmax = np.min([point[1] + r,layerShape[1]-1])

        localX = point[0] - rmin
        localY = point[1] - cmin

        maskShape = (rmax-rmin+1, cmax-cmin+1)
        localPoint = (localX,localY)
        bbox = (cmin,cmax,rmin,rmax)
        sliceIndex = indexFromBbox(bbox)

        return maskShape, localPoint, sliceIndex, bbox

def saveObjectPlot(object,i,folderpath):
    warnings.filterwarnings('ignore')
    fig = plt.figure()
    object.plotPixelsAndCentres()
    plt.savefig(path.join(folderpath,f'{pad(i,4)}_{pad(object.id,4)}.png'))
    plt.close()
    warnings.resetwarnings()

