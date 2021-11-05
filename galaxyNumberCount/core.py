from typing import Tuple
import numpy as np
from scipy.optimize.minpack import curve_fit
import scipy.stats
from scipy import ndimage
import matplotlib.pyplot as plt
from astropy.io import fits
import functools

from .colouredPrint import printC, bcolors

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
        self.backgroundStd2 = np.std(pixels[(pixels < maxFit) & (pixels > minFit)])
        self.galaxy_significance_threshold = self.backgroundStd*self.getSignificanceThresholdInStd() + self.backgroundMean

    def printBackgroundInfo(self):
        self._ensureBackground()
        
        printC(bcolors.OKBLUE, f'Background is {self.backgroundMean:.5g} +/- {self.backgroundStd:.3g} ({self.backgroundStd2:.3g})')
        printC(bcolors.OKCYAN, f'Threshold for galaxies is {self.galaxy_significance_threshold:.4g}')

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

    def blackoutAndCropBorderRegion(self,r=100,borderPixelValue=3421):        
        self.blackoutRegion = self.getEmptyMask()
        self.blackoutRegion[0,:] = True
        self.blackoutRegion[-1,:] = True
        self.blackoutRegion[:,0] = True
        self.blackoutRegion[:,-1] = True
        
        firstExpansionPixels = (self.image == borderPixelValue) | self.blackoutRegion

        changeNumberPixels = 1
        numberPixels = 0
        while changeNumberPixels != 0:
            self.blackoutRegion = ndimage.binary_dilation(self.blackoutRegion,mask=firstExpansionPixels,iterations=50)
            newNumberPixels = np.sum(self.blackoutRegion)
            changeNumberPixels = newNumberPixels - numberPixels
            numberPixels = newNumberPixels
        self.blackoutRegion = ndimage.binary_dilation(self.blackoutRegion,iterations=r)

        self.image = self.image[r:-r,r:-r].copy()
        self.blackoutRegion = self.blackoutRegion[r:-r,r:-r].copy()
        self.deadPixels = self.deadPixels[r:-r,r:-r].copy()

        self.borderFlagRegion = ndimage.binary_dilation(self.blackoutRegion,iterations=1) & ~self.blackoutRegion
        self.image[self.blackoutRegion] = 0

    def identifyObjects(self,threshold=None,expandThreshold=None,searchRegion=None):

        if threshold is None:
            self._ensureBackground()
            threshold = self.galaxy_significance_threshold
        
        if expandThreshold is None:
            self._ensureBackground()
            expandThreshold = self.galaxy_significance_threshold

        if searchRegion is None:
            searchRegion = ( slice(None,None),slice(None,None) )
        searchMask = self.getEmptyMask()
        searchMask[searchRegion] = True

        remainingCorePixels = self.image.copy()
        remainingCoreMask = (self.image > threshold)
        remainingCoreMask[~searchMask] = False
        expandMask = self.image > expandThreshold
        self.globalObjectMask = self.getEmptyMask()

        totalPixelsRemaining = np.sum(remainingCoreMask)
        print('Total core pixels:',totalPixelsRemaining)

        while totalPixelsRemaining > 0:
            print('Core pixels remaining:',totalPixelsRemaining,' '*30,end='\r')

            remainingCorePixels[np.bitwise_not(remainingCoreMask)] = 0
            firstPixel = np.unravel_index(remainingCorePixels.flatten().argmax(),self.image.shape)

            fullObjectMask, smallObjectMask, bboxObject = self._dilateObjectMask(
                expandMask, firstPixel,totalPixelsRemaining
            )
            remainingCoreMask = remainingCoreMask & np.bitwise_not(fullObjectMask)
            totalPixelsRemaining = np.sum(remainingCoreMask)
            self.globalObjectMask = self.globalObjectMask | fullObjectMask

            if smallObjectMask is not None:
                self.objects.append(AstronomicalObject(smallObjectMask,self,bboxObject))
            else:
                self.objects.append(AstronomicalObject(fullObjectMask,self))
            

    def _dilateObjectMask(self, includeMask, firstPixel, coreRemaining=None):
        
        localShape, localFirstPixel, globalSlice, bbox = squareMaskAroundPoint(firstPixel,15,self.image.shape)
        
        objectMask = np.zeros(localShape)
        objectMask[localFirstPixel] = True
        objectNumberPixels = np.sum(objectMask)
        currentIncludeMask = includeMask[globalSlice]

        changeNumberPixels = 1
        i = 0

        while changeNumberPixels != 0:
            print(f'Core pixels remaining: {coreRemaining} --- Object pixels: {objectNumberPixels}       ',end='\r')
            if i == 2:
                currentIncludeMask = includeMask
                objectMask = self._dilate_expandMask(globalSlice, objectMask)
            objectMask = ndimage.binary_dilation(objectMask,mask=currentIncludeMask,iterations=14)
            newObjectNumberPixels = np.sum(objectMask)
            changeNumberPixels = newObjectNumberPixels - objectNumberPixels
            objectNumberPixels = newObjectNumberPixels
            i = i+1

        if i < 3:
            fullObjectMask = self._dilate_expandMask(globalSlice, objectMask)
            return fullObjectMask, objectMask, bbox
        else:
            return objectMask, None, None

    def _dilate_expandMask(self, globalSlice, objectMask):
        expandedMask = self.getEmptyMask()
        expandedMask[globalSlice] = objectMask
        objectMask = expandedMask.astype(bool)
        return objectMask

    def getEmptyMask(self):
        return np.full(self.image.shape,False)

    def brightnessCount(self):
        return _FieldImageBrightnessMethodBinder(img=self)

    @functools.cache
    def dilatedGlobalObjectMask(self, iterations):
        globalObjectMask = self.globalObjectMask
        if iterations != 0:
            globalObjectMask = ndimage.binary_dilation(
                globalObjectMask,iterations=iterations
            ).astype(bool)
            
        return globalObjectMask

class _FieldImageBrightnessMethodBinder():
    """Samples the brightness of all objects not marked as discard. The method to use should be
    specified by calling a AstronomicalObject method as if it were a method of _FieldBrightnessMethodBinder,
    including appropriate arguments.
    
    Example:
        myInstance = _FieldImageBrightnessMethodBinder(myFieldImage)
        xBrights, nBrighter = myInstance.getCircularApertureBrightness(r=12,background='local')
    """
    def __init__(self,img):
        self.fieldImage = img
        self.boundName = None

    def __getattr__(self, name: str):
        """Captures the method name to use, returning self
        """
        self.boundName = name
        return self

    def __call__(self, *args, **kwargs):
        """Caputres the calling arguments to use, then iterates over all objects capturing the 
        brightness using the specified method and arguments
        """
        if self.boundName is None:
            raise Exception('Must first specify the method to use: ie. myBind.method(*args), not myBind(*args)')

        brightness_list = []
        N = len(self.fieldImage.objects)
        i = 0
        for object in self.fieldImage.objects:
            if not object.isDiscarded:
                objectMethod = getattr(object,self.boundName)
                brightness = objectMethod(*args,**kwargs)

                if brightness > 0:
                    brightness_list.append(brightness)

            print(f'Extracting object brightnesses: {i} / {N} objects complete, Method: {self.boundName}                     ',end='\r')
            i+=1

        brightness_list = sorted(brightness_list)
        brightness_list = np.array(brightness_list)
        xBrights = np.exp(np.linspace(np.log(np.min(brightness_list)),np.log(np.max(brightness_list)),500))
        nBrighter = [np.sum(brightness_list >= val) for val in xBrights]
        nBrighter = np.array(nBrighter)

        return xBrights, nBrighter

class AstronomicalObject():

    def __init__(self,objectMask,parentImageField,object_bbox=None):
        self.numberPixels = np.sum(objectMask)
        self.parentImageField =  parentImageField
        self._extractCropped(objectMask, parentImageField.image, object_bbox)
        self._computeCentreProperties()
        self._computePeakProperties()
        self.isDiscarded = False
        self.overlapsBorder = False
        self._discardInBorderRegion()

    def _discardInBorderRegion(self):
        borderMaskSlice = self.parentImageField.borderFlagRegion[indexFromBbox(self.bbox)].copy()
        if np.sum(borderMaskSlice) > 0:
            self.overlapsBorder = True
            self.isDiscarded = True

    def _extractCropped(self, objectMask, parentImage, object_bbox=None):
        self.bbox = bbox(objectMask)
        self.croppedMask = objectMask[indexFromBbox(self.bbox)].copy()
        if object_bbox is not None:
            cmin, cmax, rmin, rmax = self.bbox
            cmin_g, cmax_g, rmin_g, rmax_g = object_bbox
            self.bbox = (cmin + cmin_g, cmax + cmin_g, rmin + rmin_g, rmax + rmin_g)
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

    def getBrightnessWithoutBackground(self):
        background = self.parentImageField.backgroundMean
        return np.sum(self.croppedPixel) - self.numberPixels * background

    def getBrightnessWithoutLocalBackground(self):
        background = self.getLocalBackground()
        return np.sum(self.croppedPixel) - self.numberPixels * background

    def getCircularApertureBrightness(self,r=7,background=None,**localBackgroundKwargs):
        image = self.parentImageField.image

        if background is None:
            background = self.parentImageField.backgroundMean

        if background == 'local':
            background = self.getLocalBackground(**localBackgroundKwargs)

        sliceIndex, placementMatrix, aperture = self._getCroppedCircularAperture(r)
        includeMask = ~self._maskOtherObjectsAndEdge(r)

        pixelsInAperture = image[sliceIndex]
        brightness = np.sum(pixelsInAperture[includeMask])
        return brightness - background * np.sum(includeMask)

    @functools.cache
    def getLocalBackground(self,r=20,dilateObjectMask=3,minimumPixels=50):
        image = self.parentImageField.image

        sliceIndex, placementMatrix, aperture = self._getCroppedCircularAperture(r)
        includeMask = ~self._maskAllObjectsAndEdge(r,dilateObjectMask=dilateObjectMask)
        
        backgroundPixles = image[sliceIndex][includeMask]
        background = np.sum(backgroundPixles)/np.sum(includeMask)
        if np.sum(includeMask) < minimumPixels:
            background = self.parentImageField.backgroundMean
        return background

    @functools.cache
    def _getCroppedCircularAperture(self, r) -> Tuple:
        imageShape = self.parentImageField.image.shape
        # Why is the correct choice to use the indicies in this order here, rather than 0,1 ?
        globalCentrePoint = (round(self.globalCentreMean[1]), round(self.globalCentreMean[0]))
        maskShape, localPoint, sliceIndex, bbox = squareMaskAroundPoint(globalCentrePoint,r,imageShape)
        mask = circularMask(r)
        placementMatrix = np.zeros(maskShape)
        placementMatrix[localPoint] = 1
        if mask.shape[0] != maskShape[0] or mask.shape[1] != maskShape[1]:
            aperture = ndimage.convolve(placementMatrix,mask,mode='constant').astype(bool)
        else:
            aperture = mask.astype(bool)
        return sliceIndex,placementMatrix,aperture

    @functools.cache
    def _maskOtherObjectsAndEdge(self, r) -> np.ndarray:
        globalObjectMask = self.parentImageField.globalObjectMask
        sliceIndex, placementMatrix, aperture = self._getCroppedCircularAperture(r)

        localCentrePoint = (
            round(self.localCentreMean[1]) - (self.croppedMask.shape[0])//2,
            round(self.localCentreMean[0]) - (self.croppedMask.shape[1])//2
        )
        recroppedObjectMask = ndimage.convolve(
            placementMatrix,self.croppedMask,mode='constant',origin=np.array(localCentrePoint)
        ).astype(bool)
        mask = globalObjectMask[sliceIndex] & ~recroppedObjectMask
        mask[~aperture] = True
        return mask

    @functools.cache
    def _maskAllObjectsAndEdge(self, r, dilateObjectMask=0) -> np.ndarray:
        globalObjectMask = self.parentImageField.dilatedGlobalObjectMask(dilateObjectMask)

        sliceIndex, placementMatrix, aperture = self._getCroppedCircularAperture(r)

        mask = globalObjectMask[sliceIndex]
        mask[~aperture] = 1
        return mask

    @property
    def shape(self):
        return self.croppedMask.shape