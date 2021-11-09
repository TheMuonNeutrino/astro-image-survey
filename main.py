
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import multiprocessing
import warnings

from galaxyNumberCount.utilities import pad, clearFolder
from galaxyNumberCount import core

ROOT_PATH = path.dirname(__file__)
MOSIC_PATH = path.join(ROOT_PATH,'ExtragalacticFieldSurvey_A1.fits')
CACHE_PATH = path.join(ROOT_PATH,'FieldImageCache.pickle')

SNIPPET_IMG_FOLDER_TWIN = path.join(ROOT_PATH,'snippets_twin')
SNIPPET_IMG_FOLDER_DISCARDED = path.join(ROOT_PATH,'snippets_discarded')

USE_CACHED = True

excludeObjectIds = []

if USE_CACHED:
    #CACHE_PATH = path.join(ROOT_PATH,'FieldImageCache_ALL_0_-3std.pickle')
    CACHE_PATH = path.join(ROOT_PATH,'FieldImageCache_ALL_0_-2std.pickle'); excludeObjectIds = [69, 57, 27]
    #CACHE_PATH = path.join(ROOT_PATH,'FieldImageCache_ALL_0_-05std.pickle')

### END CONFIG ###

def saveObjectPlot_twin(object,i):
    core.saveObjectPlot(object,i,SNIPPET_IMG_FOLDER_TWIN)

def saveObjectPlot_discard(object,i):
    core.saveObjectPlot(object,i,SNIPPET_IMG_FOLDER_DISCARDED)

if __name__ == '__main__':

    if USE_CACHED:
        with open(CACHE_PATH,'rb') as file:
            img = pickle.load(file)
        img.printBackgroundInfo()
    else:
        img = core.FieldImage(MOSIC_PATH)

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

    objectsSorted = sorted(img.getIncludedObjects(),key=lambda x: x.peakMeanDistance,reverse=True)
    # clearFolder(SNIPPET_IMG_FOLDER_TWIN)
    # with multiprocessing.Pool(10) as p:
    #     p.starmap(saveObjectPlot_twin, zip(objectsSorted, range(200)))

    # objectsDiscarded = [object for object in img.objects if object.isDiscarded]
    # clearFolder(SNIPPET_IMG_FOLDER_DISCARDED)
    # with multiprocessing.Pool(10) as p:
    #     p.starmap(saveObjectPlot_discard, zip(objectsDiscarded, range(1000)))

    for object in objectsSorted[0:10]:
        object.attemptTwinSplit()

    # plt.plot(
    #     *img.brightnessCount().getBrightnessWithoutBackground(),
    #     marker='',label='Naive | Subtracted background'
    # )
    # plt.plot(
    #     *img.brightnessCount().getBrightnessWithoutLocalBackground(),
    #     marker='',label='Naive | Local background'
    # )
    # plt.plot(
    #     *img.brightnessCount().getCircularApertureBrightness(20),
    #     marker='',label='Aperture | Subtracted background'
    # )
    # plt.plot(
    #     *img.brightnessCount().getCircularApertureBrightness(
    #         20,'local',rBackground=30,dilateObjectMaskBackground=4
    #     ),
    #     marker='',label='Aperture | Local background'
    # )
    plt.xlabel('Brightness')
    plt.ylabel('Objects brighter')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

