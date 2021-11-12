
from typing import Tuple
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import functools
import itertools
import re
from copy import deepcopy
import time

from galaxyNumberCount.core import bbox, circularMask, indexFromBbox, squareMaskAroundPoint

class AstronomicalObject():

    def __init__(self,objectMask,parentImageField,object_bbox=None):
        self.numberPixels = np.sum(objectMask)
        self.parentImageField =  parentImageField
        self._extractCropped(objectMask, parentImageField.image, object_bbox)
        self._computeCentreProperties()
        self._computePeakProperties()
        self.isDiscarded = False
        self.overlapsBorder = False
        self.wasSplit = False
        self._discardInBorderRegion()
        self.id = len(parentImageField.objects)

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

    def getBrightnessWithoutLocalBackground(self,**localBackgroundKwargs):
        background = self.getLocalBackground(**localBackgroundKwargs)
        return np.sum(self.croppedPixel) - self.numberPixels * background

    def getCircularApertureBrightness(self,r=7,background=None,dilateObjectsMask=0,**localBackgroundKwargs):
        image = self.parentImageField.image

        if background is None:
            background = self.parentImageField.backgroundMean

        if background == 'local':
            background = self.getLocalBackground(**localBackgroundKwargs)

        sliceIndex, placementMatrix, aperture = self._getCroppedCircularAperture(r,r+dilateObjectsMask)
        includeMask = ~self._maskOtherObjectsAndEdge(r,dilateObjectsMask)

        pixelsInAperture = image[sliceIndex]
        brightness = np.sum(pixelsInAperture[includeMask])
        return brightness - background * np.sum(includeMask)

    @functools.cache
    def getLocalBackground(self,rBackground=20,dilateObjectMaskBackground=3,minimumPixels=50):
        image = self.parentImageField.image

        sliceIndex, placementMatrix, aperture = self._getCroppedCircularAperture(rBackground)
        includeMask = ~self._maskAllObjectsAndEdge(rBackground,dilateObjectMask=dilateObjectMaskBackground)
        
        numberOfPixelsIsSufficient = np.sum(includeMask) >= minimumPixels
        if numberOfPixelsIsSufficient:
            backgroundPixles = image[sliceIndex][includeMask]
            background = np.sum(backgroundPixles)/np.sum(includeMask)
        if not numberOfPixelsIsSufficient:
            background = self.parentImageField.backgroundMean
            
        return background

    @functools.cache
    def _getCroppedCircularAperture(self, r, d=None) -> Tuple:
        if d == None:
            d = r
        imageShape = self.parentImageField.image.shape
        # Why is the correct choice to use the indicies in this order here, rather than 0,1 ?
        globalCentrePoint = (round(self.globalCentreMean[1]), round(self.globalCentreMean[0]))
        maskShape, localPoint, sliceIndex, bbox = squareMaskAroundPoint(globalCentrePoint,d,imageShape)
        mask = circularMask(r)
        placementMatrix = np.zeros(maskShape)
        placementMatrix[localPoint] = 1
        if mask.shape[0] != maskShape[0] or mask.shape[1] != maskShape[1]:
            aperture = ndimage.convolve(placementMatrix,mask,mode='constant').astype(bool)
        else:
            aperture = mask.astype(bool)
        return sliceIndex,placementMatrix,aperture

    @functools.cache
    def _maskOtherObjectsAndEdge(self, r, dilateObjectMask = 0) -> np.ndarray:
        globalObjectMask = self.parentImageField.dilatedGlobalObjectMask(dilateObjectMask)
        sliceIndex, placementMatrix, aperture = self._getCroppedCircularAperture(r,r+dilateObjectMask)

        localCentrePoint = (
            round(self.localCentreMean[1]) - (self.croppedMask.shape[0])//2,
            round(self.localCentreMean[0]) - (self.croppedMask.shape[1])//2
        )
        
        recroppedObjectMask = ndimage.convolve(
            placementMatrix,self.croppedMask,mode='constant',origin=np.array(localCentrePoint)
        ).astype(bool)
        if dilateObjectMask != 0:
            recroppedObjectMask = ndimage.binary_dilation(recroppedObjectMask,iterations=dilateObjectMask)
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

    @property
    def rotationIndependentAspect(self):
        x = self.shape[0] / self.shape[1]
        if x >= 1:
            return x
        if x < 1:
            return 1/x

    @property
    def peakMeanDistance(self):
        dx = self.localCentreMean[0] - self.localPeak[0]
        dy = self.localCentreMean[1] - self.localPeak[1]
        return np.sqrt(dx**2 + dy**2)
    
    def plotPixelsAndCentres(self):
        plt.imshow(np.log(self.croppedPixel))
        plt.scatter(*self.localCentreMean, label='Mean Centre')
        plt.scatter(*self.localPeak, label='Peak')
        plt.legend()

    def getEmptyMask(self):
        return np.full(self.croppedPixel.shape,False)

    def attemptTwinSplit(self,promptForConfirmation = True):
        brightnessThresholds = np.trim_zeros(np.flip(np.unique(self.croppedPixel)))
        subRegions = []
        subRegionsPeak = []
        i = 0
        regionsMask = self.getEmptyMask()

        while i < len(brightnessThresholds):
            print(f"Twin split on {self.id}: Threshold {i+1} / {len(brightnessThresholds)} reached",end='\r')

            thresholdMask = self.croppedPixel >= brightnessThresholds[i]

            self._twinSplit_binaryDilationInplace(subRegions, regionsMask, thresholdMask)
            self._twinSplit_resolveConflictsInplace(subRegions, subRegionsPeak)
            regionsMask = self._twinSplit_updateRegionsMask(subRegions, regionsMask)
            existsDisconnectedPixels = self._twinSplit_addRegionIfDisconnectedInplace(
                subRegions, subRegionsPeak, regionsMask, thresholdMask
            )
            if not existsDisconnectedPixels:
                i += 1
            
        print('')

        if len(subRegions) != 1:

            if promptForConfirmation:
                self._twinSplit_userAssistedRefactor(subRegions, subRegionsPeak)
            else:
                self._twinSplit_commitSplitToParent(subRegions)

    def _twinSplit_userAssistedRefactor(self, subRegions, subRegionsPeak):
        validApprovedA = False
        oldSubRegions = None
        while not validApprovedA:
            subRegionsPixels = []
            for region in subRegions:
                subRegionsPixels.append(self.croppedPixel.copy())
                subRegionsPixels[-1][~region] = 0

            validApprovedA, validApprovedB, validApprovedC, approvedUserInput, mergeIndicies = (
                    self._twinSplit_promptUserApproval(subRegions, subRegionsPixels)
                )

            if validApprovedB:
                oldSubRegions = deepcopy(subRegions)
                subRegions[mergeIndicies[0]] = subRegions[mergeIndicies[0]] | subRegions[mergeIndicies[1]]
                del subRegions[mergeIndicies[1]]
                del subRegionsPeak[mergeIndicies[1]]
            if validApprovedC and oldSubRegions is not None:
                subRegions = oldSubRegions

        if approvedUserInput.lower() == 'y':
            self._twinSplit_commitSplitToParent(subRegions)

    def _twinSplit_commitSplitToParent(self, subRegions):
        self.isDiscarded = True
        self.wasSplit = True
        self.splitDaughterIds = []

        for region in subRegions:
            newAstronomicalObject = AstronomicalObject(region,self.parentImageField,self.bbox)
            self.parentImageField.objects.append(newAstronomicalObject)
            self.splitDaughterIds.append(newAstronomicalObject.id)

    def _twinSplit_promptUserApproval(self, subRegions, subRegionsPixels):
        plt.ion()

        fig, ax = plt.subplots(2,len(subRegions))
        plt.sca(ax[0][0])
        plt.imshow(np.log(self.croppedPixel))
        for i, region in enumerate(subRegionsPixels[0:5]):
            plt.sca(ax[1][i])
            plt.imshow(np.log(region))
        plt.draw()

        validApproved = False
        mergeIndicies = []
        while not validApproved:
            approvedUserInput = input(
                "Approve galaxy twin split (y/n) or select a pair of subregions to merge (eg. '0 1', '1 3') or undo ('undo'):"
            )
            validApprovedA = approvedUserInput.lower() in ['n','y']
            validApprovedB = re.match('^[0-9]+\s[0-9]+$',approvedUserInput)
            validApprovedC = approvedUserInput.lower() == 'undo'
            if validApprovedB:
                mergeIndicies = approvedUserInput.split()
                for i in (0,1):
                    try:
                        mergeIndicies[i] = int(mergeIndicies[i])
                        if mergeIndicies[i] >= len(subRegions) or mergeIndicies[i] < 0:
                            validApprovedB = False
                    except Exception as e:
                        validApprovedB = False
                        print(e)
                if mergeIndicies[0] == mergeIndicies[1]:
                    validApprovedB = False
            validApproved = validApprovedA or validApprovedB or validApprovedC
            
        plt.close()
        plt.ioff()
        return validApprovedA, validApprovedB, validApprovedC, approvedUserInput, mergeIndicies

    def _twinSplit_addRegionIfDisconnectedInplace(self, subRegions, subRegionsPeak, regionsMask, thresholdMask):
        disconnectedPixelMask = thresholdMask & ~regionsMask
        existsDisconnectedPixels = np.sum(disconnectedPixelMask) > 0
        if existsDisconnectedPixels:
            disconnectedPixels = self.croppedPixel.copy()
            disconnectedPixels[~disconnectedPixelMask] = 0
            regionStartIndex = np.unravel_index(
                    disconnectedPixels.flatten().argmax(),self.croppedPixel.shape
                )
            subRegions.append(self.getEmptyMask())
            subRegionsPeak.append(regionStartIndex)
            subRegions[-1][regionStartIndex] = True
        return existsDisconnectedPixels

    def _twinSplit_updateRegionsMask(self, subRegions, regionsMask):
        for region in subRegions:
            regionsMask = regionsMask | region
        return regionsMask

    def _twinSplit_resolveConflictsInplace(self, subRegions, subRegionsPeak):
        regionPairs = itertools.combinations(range(len(subRegions)),2)
        for (iA,iB) in regionPairs:
            conflicts = (subRegions[iA] & subRegions[iB])
            conflictPixelsIndecies = np.argwhere(conflicts)
            for conflictPixel in conflictPixelsIndecies:
                distanceA = np.sum( (conflictPixel-subRegionsPeak[iA])**2 )
                distanceB = np.sum( (conflictPixel-subRegionsPeak[iB])**2 )
                conflictPixel = tuple(conflictPixel.tolist())
                if distanceA >= distanceB:
                    subRegions[iA][conflictPixel] = False
                if distanceB > distanceA:
                    subRegions[iB][conflictPixel] = False

    def _twinSplit_binaryDilationInplace(self, subRegions, regionsMask, thresholdMask):
        """
        WARNING: Modifies subRegions inplace
        """
        for j, region in enumerate(subRegions):
            myStruct = np.ones((5,5))
            myStruct[0,0] = 0
            myStruct[0,-1] = 0
            myStruct[-1,0] = 0
            myStruct[-1,-1] = 0

            subRegions[j] = ndimage.binary_dilation(
                region,
                structure=myStruct,
                iterations=3,
                mask=thresholdMask & ~(regionsMask & ~region)
            )