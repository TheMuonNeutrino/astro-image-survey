import re
import numpy as np
from scipy.optimize.minpack import curve_fit
import scipy.stats
from scipy import ndimage
import matplotlib.pyplot as plt
from astropy.io import fits

from .colouredPrint import printC, bcolors

def gaussian(x,u,sigma,A):
    return A / (sigma*np.sqrt(2*np.pi)) * np.exp( -(x-u)**2 /(2*sigma**2) )

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def indexFromBbox(bbox):
    rmin, rmax, cmin, cmax = bbox
    return (
        slice(rmin,rmax+1),
        slice(cmin,cmax+1)
    )

class FieldImage():
    def __init__(self,filePath = None):
        if filePath is not None:
            self.importFits(filePath)

        self.deadPixels = self.image == 0
        self.objects = []
        self.pvalueForThreshold = 0.05

    def importFits(self,filePath):
        hdulist = fits.open(filePath)
        self.image = hdulist[0].data

    def getSignificanceThresholdInStd(self,p=None):
        number_pixels, significance_threshold_in_std = self._computeStdThreshold(p)
        return significance_threshold_in_std

    def printSignificanceThresholdInfo(self,p=None):
        if p is None:
            p = self.pvalueForThreshold
        number_pixels, significance_threshold_in_std = self._computeStdThreshold(p)
        printC('',f'Image has shape {self.image.shape}')
        printC('',f'Image has {number_pixels:.5g} pixels')
        printC(bcolors.OKCYAN,
            f'Need to be atleast {significance_threshold_in_std:.3g} standard deviations from the background value (p = {p:.3g})'
        )

    def _computeStdThreshold(self, p=None):
        if p is None:
            p = self.pvalueForThreshold
        number_pixels = np.product(self.image.shape)
        significance_threshold_in_std = scipy.stats.norm.ppf(1-1/( 1/p * number_pixels))
        return number_pixels,significance_threshold_in_std

    def computeBackground(self,minFit=3300,maxFit=3550,maxCount=500000):
        pixels = self.image.flatten()
        self.pixel_values, self.pixel_values_counts = np.unique(pixels,return_counts=True)

        filter = ( (self.pixel_values > minFit) & (self.pixel_values < maxFit) & (self.pixel_values_counts < maxCount))
        self.pixel_values_fit_background = self.pixel_values[filter]
        self.pixel_values_counts_fit_background = self.pixel_values_counts[filter]

        coeffs, cov = curve_fit(
            gaussian,
            self.pixel_values_fit_background,
            self.pixel_values_counts_fit_background,
            p0=[3430,10,4e5]
        )

        self.backgroundFitCoeffs = coeffs
        self.backgroundMean = coeffs[0]
        self.backgroundStd = coeffs[1]
        self.galaxy_significance_threshold = self.backgroundStd*self.getSignificanceThresholdInStd() + self.backgroundMean

    def printBackgroundInfo(self):
        self._ensureBackground()
        
        printC(bcolors.OKBLUE, f'Background is {self.backgroundMean:.5g} +/- {self.backgroundStd:.3g}')
        printC(bcolors.OKCYAN, f'Threshold for galaxies is then {self.galaxy_significance_threshold:.4g}')
        printC(bcolors.OKCYAN,
            f'This gives a ~{self.pvalueForThreshold*100:.1g}% chance that one point of noise will be mislabled as a galaxy'
        )

    def plotBackground(self,xlim=[3200,3600]):
        self._ensureBackground()

        x = np.linspace(
            np.min(self.pixel_values_fit_background),
            np.max(self.pixel_values_fit_background),
            300
        )
        plt.scatter(self.pixel_values,self.pixel_values_counts,ls='None',marker='.')
        plt.scatter(self.pixel_values_fit_background,self.pixel_values_counts_fit_background,ls='None',marker='.')
        plt.plot(x,gaussian(x,*self.backgroundFitCoeffs))
        plt.xlim(3200,3600)
        plt.xlabel('Pixel intensity')
        plt.ylabel('Number of pixels')
        plt.tight_layout()
        plt.show()

    def _ensureBackground(self):
        if not 'backgroundMean' in dir(self):
            self.computeBackground()

    def blackoutRectangleOrMask(self,indicies_or_mask):
        self.image[indicies_or_mask] = 0

        if not 'blackoutRegion' in dir(self):
            self.blackoutRegion = self.getEmptyMask()
        
        self.blackoutRegion[indicies_or_mask] = True

    def filterBright(self,threshold=None,expandThreshold=None):

        if threshold is None:
            self._ensureBackground()
            threshold = self.galaxy_significance_threshold
        
        if expandThreshold is None:
            self._ensureBackground()
            expandThreshold = self.galaxy_significance_threshold
            # galaxy_significance_threshold might be too high ?

        remainingCorePixels = self.image.copy()
        remainingCoreMask = self.image > threshold
        expandMask = self.image > expandThreshold

        totalPixelsRemaining = np.sum(remainingCoreMask)
        print('Total core pixels:',totalPixelsRemaining)

        while totalPixelsRemaining > 0:
            print('Core pixels remaining:',totalPixelsRemaining,' '*30,end='\r')

            remainingCorePixels[np.bitwise_not(remainingCoreMask)] = 0
            firstPixel = np.unravel_index(remainingCorePixels.flatten().argmax(),self.image.shape)

            objectMask = self._dilateObjectMask(expandMask, firstPixel,totalPixelsRemaining)
            remainingCoreMask = remainingCoreMask & np.bitwise_not(objectMask)
            totalPixelsRemaining = np.sum(remainingCoreMask)

            self.objects.append(AstronomicalObject(objectMask,self.image))
            

    def _dilateObjectMask(self, includeMask, firstPixel, coreRemaining=None):
        
        #localShape, localFirstPixel, globalBox = self._maskAroundPoint(firstPixel,20,self.image.shape)
        
        objectMask = self.getEmptyMask()
        objectMask[firstPixel] = True
        objectNumberPixels = np.sum(objectMask)

        changeNumberPixels = 1

        while changeNumberPixels != 0:
            print(f'Core pixels remaining: {coreRemaining} --- Object pixels: {objectNumberPixels}       ',end='\r')
            objectMask = ndimage.binary_dilation(objectMask,mask=includeMask,iterations=10)
            newObjectNumberPixels = np.sum(objectMask)
            changeNumberPixels = newObjectNumberPixels - objectNumberPixels
            objectNumberPixels = newObjectNumberPixels
        return objectMask

    def _maskAroundPoint(self,point,r,layerShape):
        rmin = np.max(point[0] - r,0)
        rmax = np.min(point[0] + r,layerShape[0])
        cmin = np.max(point[1] - r,0)
        cmax = np.min(point[1] + r,layerShape[1])

        localX = point[0] - rmin
        localY = point[1] - cmin

        return (rmax, cmax), (localY, localX), indexFromBbox((rmin,rmax,cmin,cmax))

    def getEmptyMask(self):
        return np.full(self.image.shape,False)

    def brightnessCount(self):
        return _FieldImageBrightnessMethodBinder(img=self)

class _FieldImageBrightnessMethodBinder():
    def __init__(self,img):
        self.fieldImage = img
        self.boundName = None

    def __getattr__(self, name: str):
        self.boundName = name
        return self

    def __call__(self, *args, **kwargs):
        if self.boundName is None:
            raise Exception('Must first specify the method to use: ie. myBind.method(*args), not myBind(*args)')

        brightness_list = []
        for object in self.fieldImage.objects:
            objectMethod = getattr(object,self.boundName)
            brightness = objectMethod(*args,**kwargs)

            if brightness > 0:
                brightness_list.append(brightness)

        brightness_list = sorted(brightness_list)
        brightness_list = np.array(brightness_list)
        xbrights = np.exp(np.linspace(np.log(np.min(brightness_list)-1),np.log(np.max(brightness_list)),500))
        nBrighter = [np.sum(brightness_list >= val) for val in xbrights]
        nBrighter = np.array(nBrighter)

        return xbrights, nBrighter

class AstronomicalObject():
    def __init__(self,objectMask,parentImage):
        self.numberPixels = np.sum(objectMask)
        self._extractCropped(objectMask, parentImage)
        self._computeCentreProperties()
        self._computePeakProperties()

    def _extractCropped(self, objectMask, parentImage):
        self.bbox = bbox(objectMask)
        self.croppedMask = objectMask[indexFromBbox(self.bbox)].copy()
        self.croppedPixel = parentImage[indexFromBbox(self.bbox)].copy()
        self.croppedPixel[np.bitwise_not(self.croppedMask)] = 0

    def _computeCentreProperties(self):
        self.brightDistribution = [None,None]
        self.localCentreMean = [None,None]

        for axis,sumAxis in ((0,1),(1,0)):
            self.brightDistribution[axis] = np.sum(self.croppedPixel,axis=sumAxis)
            self.localCentreMean[sumAxis] = np.average(
                np.arange(self.croppedPixel.shape[axis]),
                weights=self.brightDistribution[axis]
            )
        
        self.globalCentreMean = self._localPointToGlobal(self.localCentreMean)

    def _computePeakProperties(self):
        self.localPeak = [None,None]

        for axis in (0,1):
            maxPx = np.max(self.croppedPixel,axis=axis)
            self.localPeak[axis] = np.argmax(maxPx)

        self.globalPeak = self._localPointToGlobal(self.localPeak)

    def _localPointToGlobal(self,point):
        return [
            point[0] + self.bbox[0],
            point[1] + self.bbox[2]
        ]

    def getNaiveBrightness(self):
        return np.sum(self.croppedPixel)

    def getBrightnessWithoutBackground(self,background):
        return np.sum(self.croppedPixel) - self.numberPixels * background