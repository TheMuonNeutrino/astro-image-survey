
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from galaxyNumberCount.astronomicalObjectClass import AstronomicalObject
import multiprocessing

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
        img.printBackgroundInfo()
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
        with open(CACHE_PATH,'wb') as file:
            pickle.dump(img,file)

    for object in img.objects:
        if object.id in excludeObjectIds:
            object.isDiscarded = True

    print('Saving twins snippets')
    objectsTwins = sorted(img.getIncludedObjects(),key=lambda x: x.peakMeanDistance,reverse=True)
    objectsTwins = [object for object in objectsTwins if (
        np.min(object.shape) > 5 and object.peakMeanDistance > 2
    )]
    if SAVE_SNIPPETS:
        clearFolder(SNIPPET_IMG_FOLDER_TWIN)
        with multiprocessing.Pool(10) as p:
            p.starmap(saveObjectPlot_twin, zip(objectsTwins, range(30)))

    print('Saving discarded snippets')
    objectsDiscarded = [object for object in img.objects if object.isDiscarded]
    if SAVE_SNIPPETS:
        clearFolder(SNIPPET_IMG_FOLDER_DISCARDED)
        with multiprocessing.Pool(2) as p:
            p.starmap(saveObjectPlot_discard, zip(objectsDiscarded, range(1000)))
    
    print('Saving largest objects snippets')
    objectsLargest = sorted(img.getIncludedObjects(),key=lambda x: np.max(x.shape),reverse=True)
    if SAVE_SNIPPETS:
        clearFolder(SNIPPET_IMG_FOLDER_LARGEST)
        with multiprocessing.Pool(2) as p:
            p.starmap(saveObjectPlot_largest, zip(objectsLargest, range(30)))

    print(f'There are {len(objectsTwins)} potential twins to process')

    for object in objectsTwins:
        object: AstronomicalObject = object
        object.attemptTwinSplit(promptForConfirmation=False)

    print("Finished splitting twins")

    plt.close('all')

    plt.plot(
        *img.brightnessCount().getBrightnessWithoutBackground(),
        marker='',label='Naive | Subtracted background'
    )
    plt.plot(
        *img.brightnessCount().getBrightnessWithoutLocalBackground(),
        marker='',label='Naive | Local background'
    )
    plt.plot(
        *img.brightnessCount().getCircularApertureBrightness(20),
        marker='',label='Aperture | Subtracted background'
    )
    plt.plot(
        *img.brightnessCount().getCircularApertureBrightness(
            20,'local',rBackground=30,dilateObjectMaskBackground=4
        ),
        marker='',label='Aperture | Local background'
    )
    plt.xlabel('Brightness')
    plt.ylabel('Objects brighter')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()