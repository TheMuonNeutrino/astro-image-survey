import numpy as np
from scipy.optimize.minpack import curve_fit
import scipy.stats
from scipy import ndimage
import matplotlib.pyplot as plt
from astropy.io import fits
import functools

from galaxyNumberCount.astronomicalObjectClass import AstronomicalObject
from galaxyNumberCount.core import gaussian, squareMaskAroundPoint

from .utilities import printC, bcolors

class FieldImage():
    def __init__(self,filePath = None):
        if filePath is not None:
            self.importFits(filePath)

        self.deadPixels = self.image == 0
        self.objects = []
        self.pvalueForThreshold = 0.05

    def importFits(self,filePath):
        with fits.open(filePath) as hdulist:
            self.image = hdulist[0].data.copy()
            self.header = hdulist[0].header.copy()

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

    def blackoutAndCropBorderRegion(self,r=100,borderPixelValue=3421,p_border=1):        
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

        self.borderFlagRegion = ndimage.binary_dilation(self.blackoutRegion,iterations=p_border) & ~self.blackoutRegion
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

    def brightnessCountPlot(self):
        return _FieldImageBrightnessMethodBinder(img=self,callback=self._brightnessCountPlot_callback)

    def _brightnessCountPlot_callback(self,brightness_list):
        xBrights = np.exp(np.linspace(np.log(np.min(brightness_list)),np.log(np.max(brightness_list)),500))
        nBrighter = [np.sum(brightness_list >= val) for val in xBrights]
        nBrighter = np.array(nBrighter)
        return xBrights, nBrighter

    def magnitudeCountPlot(self):
        return _FieldImageBrightnessMethodBinder(self,self._magnitueCountPlot_callback)

    def _magnitueCountPlot_callback(self,brightness_list):
        xBrights, nBrighter = self._brightnessCountPlot_callback(brightness_list)
        xMagnitude = self.header['MAGZPT'] - 2.5 * np.log10(xBrights)
        return xMagnitude, nBrighter
    
    def magnitudeCountFit(self):
        return _FieldImageBrightnessMethodBinder(self,self._magnitueCountFit_callback)

    def _magnitueCountFit_callback(self,brightness_list):
        xBrights = brightness_list
        nBrighter = [np.sum(brightness_list >= val) for val in xBrights]
        nBrighter = np.array(nBrighter)
        xMagnitude = self.header['MAGZPT'] - 2.5 * np.log10(xBrights)
        return xMagnitude, nBrighter

    @functools.cache
    def dilatedGlobalObjectMask(self, iterations):
        globalObjectMask = self.globalObjectMask
        if iterations != 0:
            globalObjectMask = ndimage.binary_dilation(
                globalObjectMask,iterations=iterations
            ).astype(bool)
            
        return globalObjectMask

    def getIncludedObjects(self):
        return [object for object in self.objects if not object.isDiscarded]

    def seperateTwins(self):
        objectsTwins = sorted(self.getIncludedObjects(),key=lambda x: x.peakMeanDistance,reverse=True)
        objectsTwins = [object for object in objectsTwins if (
            np.min(object.shape) > 5 and object.peakMeanDistance > 2
        )]
        print(f'There are {len(objectsTwins)} potential twins to process')

        for object in objectsTwins:
            object: AstronomicalObject = object
            object.attemptTwinSplit(promptForConfirmation=False)

        print("Finished splitting twins")
        plt.close('all')

class _FieldImageBrightnessMethodBinder():
    """Samples the brightness of all objects not marked as discard. The method to use should be
    specified by calling a AstronomicalObject method as if it were a method of _FieldBrightnessMethodBinder,
    including appropriate arguments.
    
    Example:
        myInstance = _FieldImageBrightnessMethodBinder(myFieldImage,myFieldImage.myCallback)
        xBrights, nBrighter = myInstance.getCircularApertureBrightness(r=12,background='local')
    """
    def __init__(self,img,callback):
        self.fieldImage = img
        self.boundName = None
        self.callback = callback

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

            i+=1
            print(f'Extracting object brightnesses: {i} / {N} objects complete, Method: {self.boundName}                     ',end='\r')

        brightness_list = sorted(brightness_list)
        brightness_list = np.array(brightness_list)
        

        print('')

        return self.callback(brightness_list)