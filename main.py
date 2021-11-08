
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

SNIPPET_IMG_FOLDER = path.join(ROOT_PATH,'snippets_aspect')

USE_CACHED = True

if USE_CACHED:
    CACHE_ALL_MINUS_3_STD = path.join(ROOT_PATH,'FieldImageCache_ALL_0_-3std.pickle')
    CACHE_ALL_MINUS_2_STD = path.join(ROOT_PATH,'FieldImageCache_ALL_0_-2std.pickle')
    CACHE_PATH = CACHE_ALL_MINUS_2_STD

### END CONFIG ###

def saveObjectPlot(object,i):
    warnings.filterwarnings('ignore')
    fig = plt.figure()
    object.plotPixelsAndCentres()
    plt.savefig(path.join(SNIPPET_IMG_FOLDER,f'{pad(i,4)}_{pad(object.id,4)}.png'))
    plt.close()
    warnings.resetwarnings()

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
            img.galaxy_significance_threshold - 2 * img.backgroundStd,
            #(slice(0,4000), slice(0,900))
        )
        with open(CACHE_PATH,'wb') as file:
            pickle.dump(img,file)

    i = 0
    for object in img.objects:
        object.id = i
        i += 1
        object: core.AstronomicalObject = object
        object.overlapsBorder = False
        object._discardInBorderRegion()
        #object.isDiscarded = False
        
        if object.shape[0] > 50 or object.shape[1] > 50:
            object.isDiscarded = True

    objectsSorted = sorted(img.getIncludedObjects(),key=lambda x: x.peakMeanDistance,reverse=True)
    objectsSorted = [object for object in objectsSorted if np.min(object.shape) > 1]

    clearFolder(SNIPPET_IMG_FOLDER)

    with multiprocessing.Pool(10) as p:
        p.starmap(saveObjectPlot, zip(objectsSorted, range(200)))

    exit()

    image = img.image.copy()
    image[~img.globalObjectMask] = 0

    plt.plot(
        *img.brightnessCount().getBrightnessWithoutBackground(),
        marker='',label='Naive | Subtracted background'
    )
    plt.plot(
        *img.brightnessCount().getBrightnessWithoutLocalBackground(),
        marker='',label='Naive | Local background'
    )
    plt.plot(
        *img.brightnessCount().getCircularApertureBrightness(15),
        marker='',label='Aperture | Subtracted background'
    )
    plt.plot(
        *img.brightnessCount().getCircularApertureBrightness(15,'local'),
        marker='',label='Aperture | Local background'
    )
    plt.xlabel('Brightness')
    plt.ylabel('Objects brighter')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

