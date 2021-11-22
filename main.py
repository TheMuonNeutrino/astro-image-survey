
from os import path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pickle
from galaxyNumberCount.astronomicalObjectClass import AstronomicalObject
import multiprocessing
import scipy.stats

from galaxyNumberCount import core, plottools
from galaxyNumberCount.fieldImageClass import FieldImage

from galaxyNumberCount.utilities import bcolors, clearFolder, ensureFolder, printC

ROOT_PATH = path.dirname(__file__)
MOSIC_PATH = path.join(ROOT_PATH,'ExtragalacticFieldSurvey_A1.fits')
CACHE_PATH = path.join(ROOT_PATH,'FieldImageCache.pickle')
CACHE_PATH_NUMBER_COUNTS = path.join(ROOT_PATH,'NumberCountsCache.pickle')

SNIPPET_IMG_FOLDER_TWIN = path.join(ROOT_PATH,'snippets_twin')
SNIPPET_IMG_FOLDER_DISCARDED = path.join(ROOT_PATH,'snippets_discarded')
SNIPPET_IMG_FOLDER_LARGEST = path.join(ROOT_PATH, 'snippets_largest')
SNIPPET_IMG_FOLDER_BRIGHT_DISCREP_PATH = path.join(ROOT_PATH,'snippets_brightness_discrepancy')

SAVE_SNIPPETS = False
USE_CACHED = True
USE_CACHED_NUMBER_COUNTS = False
ANALYSE_BRIGHT_DISCREPANCY = False
PLOT_BK = False
PLOT_BK_DISCREP = False
#PLOT_MAGNITUDE_DISCREP = TRUE
SPLIT_TWINS = True
SAVE_SPLIT_TWINS = True

local_bk_param = {'rBackground':30,'dilateObjectMaskBackground':6, 'minimumPixels':20}

excludeObjectIds = [0]

if USE_CACHED:
    excludeObjectIds = [0,1,3,4,13,11,15,17,7,58,27,70,]
    pass

### END CONFIG ###

def saveObjectPlot_twin(object,i):
    plottools.saveTwinPlot(object,i,SNIPPET_IMG_FOLDER_TWIN)

def saveObjectPlot_discard(object,i):
    plottools.saveObjectPlot(object,i,SNIPPET_IMG_FOLDER_DISCARDED)

def saveObjectPlot_largest(object,i):
    plottools.saveObjectPlot(object,i,SNIPPET_IMG_FOLDER_LARGEST)

def excludeFlagged(excludeObjectIds, img: FieldImage, peformExclusionsInBorderRegion = False):
    for object in img.objects:
        if object.id in excludeObjectIds:
            object.isDiscarded = True
        if np.max(object.shape) > 200:
            object.isDiscarded = True
        if peformExclusionsInBorderRegion:
            object.discardInBorderRegion()

if __name__ == '__main__':

    if USE_CACHED:
        with open(CACHE_PATH,'rb') as file:
            img: FieldImage = pickle.load(file)
        
        if not img.twinSeperationWasRun and SPLIT_TWINS:
            excludeFlagged(excludeObjectIds, img)
            img.seperateTwins(minSep=1.5)
            excludeFlagged(excludeObjectIds, img,True)

            if SAVE_SPLIT_TWINS:
                with open(CACHE_PATH,'wb') as file:
                    pickle.dump(img,file)

        img.printSignificanceThresholdInfo()
        img.printBackgroundInfo()
        
    else:
        img = FieldImage(MOSIC_PATH)

        img.blackoutAndCropBorderRegion()

        print(f'Image contains {np.sum(img.deadPixels)} dead pixels')

        img.printSignificanceThresholdInfo()
        img.printBackgroundInfo()

        bkThresholdDiff = img.galaxy_significance_threshold - img.backgroundMean

        img.identifyObjects(
            img.galaxy_significance_threshold - 0 * bkThresholdDiff,
            img.galaxy_significance_threshold - 0.5 * bkThresholdDiff,
            #(slice(0,4000), slice(0,900))
        )

        excludeFlagged(excludeObjectIds, img)

        with open(CACHE_PATH,'wb') as file:
            pickle.dump(img,file)
    
    if SAVE_SNIPPETS:
        objectsTwins = [object for object in img.objects if object.wasSplit]
        print('Saving twins snippets')
        clearFolder(SNIPPET_IMG_FOLDER_TWIN)
        with multiprocessing.Pool(10) as p:
            p.starmap(saveObjectPlot_twin, zip(objectsTwins, range(300)))

        objectsDiscarded = [object for object in img.objects if (object.isDiscarded and not object.wasSplit)]
        print('Saving discarded snippets')
        clearFolder(SNIPPET_IMG_FOLDER_DISCARDED)
        with multiprocessing.Pool(2) as p:
            p.starmap(saveObjectPlot_discard, zip(objectsDiscarded, range(1000)))
    
        objectsLargest = sorted(img.getIncludedObjects(),key=lambda x: np.max(x.shape),reverse=True)
        print('Saving largest objects snippets')
        clearFolder(SNIPPET_IMG_FOLDER_LARGEST)
        with multiprocessing.Pool(10) as p:
            p.starmap(saveObjectPlot_largest, zip(objectsLargest, range(300)))

    if USE_CACHED_NUMBER_COUNTS:
        with open(CACHE_PATH_NUMBER_COUNTS,'rb') as file:
            numberCounts = pickle.load(file)

    else:
        extractionFunc = img.magnitudeCountBinned
        rFunc = lambda w, h: int( np.max([3,np.max([w,h])*0.8]) // 1 )
        numberCounts = {
            # 'Naive | Subtracted background': extractionFunc().getBrightnessWithoutBackground(),
            'Naive | Local background': extractionFunc().getBrightnessWithoutLocalBackground(
                **local_bk_param
            ),
            # 'Aperture | Subtracted background': extractionFunc().getCircularApertureBrightness(
            #     rFunc
            # ),
            'Aperture | Local background': extractionFunc().getCircularApertureBrightness(
                rFunc,'local',**local_bk_param
            )
        }
        with open(CACHE_PATH_NUMBER_COUNTS,'wb') as file:
            pickle.dump(numberCounts,file)

    background = img.image.copy()
    minBK = img.backgroundMean-3*img.backgroundStd
    background[img.globalObjectMask] = minBK
    background[background < minBK] = minBK

    if PLOT_BK:
        plt.imshow(background)
        plt.show()

    for object in img.getIncludedObjects():
        object.brightDiscrepancy = (
            -np.log(object.getBrightnessWithoutLocalBackground(**local_bk_param)) - 
            -np.log(object.getCircularApertureBrightness(rFunc,**local_bk_param))
        )
    objectsDiscrepancy = sorted(img.getIncludedObjects(),key=lambda x: x.brightDiscrepancy,reverse=False)

    if ANALYSE_BRIGHT_DISCREPANCY:
        ensureFolder(SNIPPET_IMG_FOLDER_BRIGHT_DISCREP_PATH)
        for i, object in enumerate(objectsDiscrepancy):
            if i > 1000:
                break
            if np.max(object.shape) < 20:
                r = rFunc(*object.shape)
                sliceIndex, placementMatrix, aperture = object._getCroppedCircularAperture(r,r)
                includeMask = ~object._maskOtherObjectsAndEdge(r,0)
                pixelsInAperture = img.image[sliceIndex]
                #pixelsInAperture[0,0] = 0
                fig, axs = plt.subplots(2, 2)
                axs: Tuple[plt.Axes] = axs.flatten()
                axs[0].imshow(object.croppedPixel)
                axs[1].imshow(includeMask)
                axs[2].imshow(pixelsInAperture)
                axs[3].imshow(background[sliceIndex])
                axs[0].set_title(f"m: {-np.log(object.getBrightnessWithoutLocalBackground(**local_bk_param)):.3g}")
                axs[2].set_title(f"m: {-np.log(object.getCircularApertureBrightness(rFunc,**local_bk_param)):.3g}")
                axs[1].set_title(f"id: {object.id} \n pos: {object.globalPeak[0]+100},{object.globalPeak[1]+100}")
                axs[3].set_title(f"bk: {object.getLocalBackground(**local_bk_param)-img.backgroundMean:.3g}")
                plt.tight_layout()
                plt.savefig(path.join(SNIPPET_IMG_FOLDER_BRIGHT_DISCREP_PATH,f'{core.pad(i,4)}_{core.pad(object.id,4)}.png'))
                plt.close()

    if PLOT_BK_DISCREP:
        discrepancy = np.array([object.brightDiscrepancy for object in objectsDiscrepancy])
        bk_offset = np.array([object.getLocalBackground(**local_bk_param)-img.backgroundMean for object in objectsDiscrepancy])
        mask = ~np.isnan(discrepancy) & ~np.isnan(bk_offset)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(discrepancy[mask],bk_offset[mask])
        plt.scatter(discrepancy,bk_offset,marker='.')
        printC(bcolors.WARNING,'Bk_discrep:',slope,intercept,r_value)
        plt.plot(discrepancy,slope*discrepancy + intercept,color='C1')
        plt.xlabel('Magnitude discrepancy [naive() - aperture()]')
        plt.ylabel('Local background offset from global bk')
        plt.show()

    for key, (xMagnitude, nBright) in numberCounts.items():
        nBrighter = np.cumsum(nBright)
        nBrighter_err = np.sqrt(nBright)
        indicies = (xMagnitude >= 12) & (xMagnitude < 16)
        xMagnitudeFit = xMagnitude[indicies]
        nBrighterFit = nBrighter[indicies]
        result = scipy.stats.linregress(xMagnitudeFit, np.log(nBrighterFit))
        intercept_std_err = result.intercept_stderr
        slope, intercept, r_value, p_value, std_err = result
        r_squared = r_value**2

        printC(bcolors.OKGREEN,f'{key} | r^2: {r_squared} | p: {p_value}')
        printC(bcolors.OKGREEN,f'    m: {slope:.5g} +/- {std_err:.3g}')
        printC(bcolors.OKGREEN,f'    c: {intercept:.5g} +/- {intercept_std_err:.3g}')
        
        plt.errorbar(xMagnitude,nBrighter,yerr=nBrighter_err,marker='.',ls='',label=key,capsize=3)
        plt.plot(xMagnitudeFit,np.exp(slope*xMagnitudeFit + intercept),marker='',label=key + " | Fit")

    plt.xlabel('Magnitude')
    plt.ylabel('Number of Objects Brighter')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()