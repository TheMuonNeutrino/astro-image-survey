
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from galaxyNumberCount.astronomicalObjectClass import AstronomicalObject
import multiprocessing
import scipy.stats

from galaxyNumberCount import core
from galaxyNumberCount.fieldImageClass import FieldImage

from galaxyNumberCount.utilities import bcolors, clearFolder, printC

ROOT_PATH = path.dirname(__file__)
MOSIC_PATH = path.join(ROOT_PATH,'ExtragalacticFieldSurvey_A1.fits')
CACHE_PATH = path.join(ROOT_PATH,'FieldImageCache.pickle')
CACHE_PATH_NUMBER_COUNTS = path.join(ROOT_PATH,'NumberCountsCache.pickle')

SNIPPET_IMG_FOLDER_TWIN = path.join(ROOT_PATH,'snippets_twin')
SNIPPET_IMG_FOLDER_DISCARDED = path.join(ROOT_PATH,'snippets_discarded')
SNIPPET_IMG_FOLDER_LARGEST = path.join(ROOT_PATH, 'snippets_largest')

SAVE_SNIPPETS = False
USE_CACHED = True
USE_CACHED_NUMBER_COUNTS = False

excludeObjectIds = []

if USE_CACHED:
    # CACHE_PATH = "FieldImageCache_-3_partial.pickle"; excludeObjectIds = [0, 2, 3, 6]
    CACHE_PATH = "FieldImageCache_full2.pickle"; excludeObjectIds = [0, 1, 2, 3, 4, 7, 15, 27, 70, 58, 13]
    pass

### END CONFIG ###

def saveObjectPlot_twin(object,i):
    core.saveObjectPlot(object,i,SNIPPET_IMG_FOLDER_TWIN)

def saveObjectPlot_discard(object,i):
    core.saveObjectPlot(object,i,SNIPPET_IMG_FOLDER_DISCARDED)

def saveObjectPlot_largest(object,i):
    core.saveObjectPlot(object,i,SNIPPET_IMG_FOLDER_LARGEST)

def excludeFlagged(excludeObjectIds, img, peformExclusionsInBorderRegion = False):
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
            img = pickle.load(file)
        
        excludeFlagged(excludeObjectIds, img,True)

        # with open(CACHE_PATH,'wb') as file:
        #     pickle.dump(img,file)
    else:
        img = FieldImage(MOSIC_PATH)

        img.blackoutAndCropBorderRegion()

        print(f'Image contains {np.sum(img.deadPixels)} dead pixels')

        img.printSignificanceThresholdInfo()
        img.printBackgroundInfo()

        img.identifyObjects(
            img.galaxy_significance_threshold + 0 * img.backgroundStd,
            img.galaxy_significance_threshold - 3 * img.backgroundStd,
            #(slice(0,4000), slice(0,900))
        )

        excludeFlagged(excludeObjectIds, img)
        img.seperateTwins(minSep=1.5)
        excludeFlagged(excludeObjectIds, img,True)

        with open(CACHE_PATH,'wb') as file:
            pickle.dump(img,file)
    

    if SAVE_SNIPPETS:
        objectsTwins = [object for object in img.objects if object.wasSplit]
        print('Saving twins snippets')
        clearFolder(SNIPPET_IMG_FOLDER_TWIN)
        with multiprocessing.Pool(10) as p:
            p.starmap(saveObjectPlot_twin, zip(objectsTwins, range(30)))

        objectsDiscarded = [object for object in img.objects if object.isDiscarded]
        print('Saving discarded snippets')
        clearFolder(SNIPPET_IMG_FOLDER_DISCARDED)
        with multiprocessing.Pool(2) as p:
            p.starmap(saveObjectPlot_discard, zip(objectsDiscarded, range(1000)))
    
        objectsLargest = sorted(img.getIncludedObjects(),key=lambda x: np.max(x.shape),reverse=True)
        print('Saving largest objects snippets')
        clearFolder(SNIPPET_IMG_FOLDER_LARGEST)
        with multiprocessing.Pool(10) as p:
            p.starmap(saveObjectPlot_largest, zip(objectsLargest, range(300)))

    img: FieldImage = img

    if USE_CACHED_NUMBER_COUNTS:
        with open(CACHE_PATH_NUMBER_COUNTS,'rb') as file:
            numberCounts = pickle.load(file)

    else:
        extractionFunc = img.magnitudeCountBinned
        numberCounts = {
            'Naive | Subtracted background': extractionFunc().getBrightnessWithoutBackground(),
            'Naive | Local background': extractionFunc().getBrightnessWithoutLocalBackground(
                rBackground=30,dilateObjectMaskBackground=6,minimumPixels=50
            ),
            'Aperture | Subtracted background': extractionFunc().getCircularApertureBrightness(
                12,dilateObjectsMask=4
            ),
            'Aperture | Local background': extractionFunc().getCircularApertureBrightness(
                12,'local',dilateObjectsMask=4,rBackground=30,dilateObjectMaskBackground=6
            )
        }
        with open(CACHE_PATH_NUMBER_COUNTS,'wb') as file:
            pickle.dump(numberCounts,file)

    for key, (xMagnitude, nBright) in numberCounts.items():
        nBrighter = np.cumsum(nBright)
        nBrighter_err = np.sqrt(nBright)
        indicies = (xMagnitude >= 4) & (xMagnitude < 8)
        #indicies = slice(None,None)
        xMagnitudeFit = xMagnitude[indicies]
        nBrighterFit = nBrighter[indicies]
        result = scipy.stats.linregress(xMagnitudeFit, np.log(nBrighterFit))
        intercept_std_err = result.intercept_stderr
        slope, intercept, r_value, p_value, std_err = result
        r_squared = r_value**2

        printC(bcolors.OKGREEN,f'{key} | r^2: {r_squared}')
        printC(bcolors.OKGREEN,f'    m: {slope:.5g} +/- {std_err:.3g}')
        printC(bcolors.OKGREEN,f'    c: {intercept:.5g} +/- {intercept_std_err:.3g}')
        
        plt.errorbar(xMagnitude,nBrighter,yerr=nBrighter_err,marker='.',ls='',label=key)
        plt.plot(xMagnitudeFit,np.exp(slope*xMagnitudeFit + intercept),marker='',label=key + " | Fit")

    plt.xlabel('Magnitude')
    plt.ylabel('Number of Objects Brighter')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()