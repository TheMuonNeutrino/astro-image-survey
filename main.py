
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from galaxyNumberCount.astronomicalObjectClass import AstronomicalObject
import multiprocessing
from astropy.io import fits

from galaxyNumberCount import core
from galaxyNumberCount.fieldImageClass import FieldImage
import time

from galaxyNumberCount.utilities import clearFolder

ROOT_PATH = path.dirname(__file__)
MOSIC_PATH = path.join(ROOT_PATH,'ExtragalacticFieldSurvey_A1.fits')
CACHE_PATH = path.join(ROOT_PATH,'FieldImageCache.pickle')

SNIPPET_IMG_FOLDER_TWIN = path.join(ROOT_PATH,'snippets_git_twin')
SNIPPET_IMG_FOLDER_DISCARDED = path.join(ROOT_PATH,'snippets_discarded')
SNIPPET_IMG_FOLDER_LARGEST = path.join(ROOT_PATH, 'snippets_largest')

SAVE_SNIPPETS = False
USE_CACHED = True

excludeObjectIds = []

if USE_CACHED:
    CACHE_PATH = "FieldImageCache_-3_partial.pickle"; excludeObjectIds = [0, 2, 3, 6]
    CACHE_PATH = "FieldImageCache_-3_full.pickle"; excludeObjectIds = [1, 3, 4, 13, 11, 7, 15, 13, 27, 57, 69]

### END CONFIG ###

def saveObjectPlot_twin(object,i):
    core.saveObjectPlot(object,i,SNIPPET_IMG_FOLDER_TWIN)

def saveObjectPlot_discard(object,i):
    core.saveObjectPlot(object,i,SNIPPET_IMG_FOLDER_DISCARDED)

def saveObjectPlot_largest(object,i):
    core.saveObjectPlot(object,i,SNIPPET_IMG_FOLDER_LARGEST)

if __name__ == '__main__':

    if USE_CACHED:
        with open(CACHE_PATH,'rb') as file:
            img = pickle.load(file)
        
        # for object in img.objects:
        #     if object.id in excludeObjectIds:
        #         object.isDiscarded = True
        # img.header = fits.getheader(MOSIC_PATH,0)
        # img.seperateTwins()

        # with open(CACHE_PATH,'wb') as file:
        #     pickle.dump(img,file)
    else:
        img = FieldImage(MOSIC_PATH)

        # for key in img.header.keys():
        #     if str(key).strip() != '':
        #         print(key, "|", " ".join(str(img.header[key]).split()))

        img.blackoutAndCropBorderRegion()

        print(f'Image contains {np.sum(img.deadPixels)} dead pixels')

        img.printSignificanceThresholdInfo()
        img.printBackgroundInfo()

        img.identifyObjects(
            img.galaxy_significance_threshold + 0 * img.backgroundStd,
            img.galaxy_significance_threshold - 3 * img.backgroundStd,
            #(slice(0,4000), slice(0,900))
        )
        with open(CACHE_PATH,'wb') as file:
            pickle.dump(img,file)

    for object in img.objects:
        if object.id in excludeObjectIds:
            object.isDiscarded = True

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
        with multiprocessing.Pool(2) as p:
            p.starmap(saveObjectPlot_largest, zip(objectsLargest, range(30)))

    img: FieldImage = img

    plt.plot(
        *img.magnitudeCountPlot().getBrightnessWithoutBackground(),
        marker='',label='Naive | Subtracted background'
    )
    plt.plot(
        *img.magnitudeCountPlot().getBrightnessWithoutLocalBackground(
            rBackground=30,dilateObjectMaskBackground=4,minimumPixels=50
        ),
        marker='',label='Naive | Local background'
    )
    plt.plot(
        *img.magnitudeCountPlot().getCircularApertureBrightness(12,dilateObjectsMask=3),
        marker='',label='Aperture | Subtracted background'
    )
    plt.plot(
        *img.magnitudeCountPlot().getCircularApertureBrightness(
            12,'local',dilateObjectsMask=3,rBackground=30,dilateObjectMaskBackground=4
        ),
        marker='',label='Aperture | Local background'
    )
    plt.xlabel('Magnitude')
    plt.ylabel('Objects brighter')
    plt.yscale('log')
    #plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()