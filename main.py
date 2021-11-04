
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pickle

from galaxyNumberCount.colouredPrint import printC, bcolors
from galaxyNumberCount import core

ROOT_PATH = path.dirname(__file__)
MOSIC_PATH = path.join(ROOT_PATH,'ExtragalacticFieldSurvey_A1.fits')
CACHE_PATH = path.join(ROOT_PATH,'FieldImageCache.pickle')

USE_CACHED = False

if USE_CACHED:
    CACHE_1000_4000_100_0 = path.join(ROOT_PATH,'FieldImageCache_1000_4000_100_0.pickle')
    CACHE_1000_4000_10_MINUS10 = path.join(ROOT_PATH,'FieldImageCache_1000_4000_10_-10.pickle')
    #CACHE_PATH = CACHE_1000_4000_10_MINUS10

### END CONFIG ###

if USE_CACHED:
    with open(CACHE_PATH,'rb') as file:
        img = pickle.load(file)
    img.printBackgroundInfo()
else:
    img = core.FieldImage(MOSIC_PATH)
    img.blackoutRectangleOrMask((slice(1038,3094),slice(2482,-1)))

    img.pvalueForThreshold = 0.005
    img.printSignificanceThresholdInfo()
    img.printBackgroundInfo()

    img.blackoutRectangleOrMask((slice(0,-1),slice(50,-1)))
    img.blackoutRectangleOrMask((slice(4000,-1),slice(0,-1)))

    img.filterBright(img.galaxy_significance_threshold+10,img.galaxy_significance_threshold-10)
    with open(CACHE_PATH,'wb') as file:
        pickle.dump(img,file)


for object in img.objects:
    if object.numberPixels > 1 and object.getNaiveBrightness() == 0:
        plt.imshow(object.croppedPixel)
        plt.scatter(*object.localCentreMean)
        plt.scatter(*object.localPeak)
        plt.show()

plt.plot(
    *img.brightnessCount().getBrightnessWithoutBackground(img.backgroundMean),
    marker='',label='Subtracted background'
)
plt.plot(
    *img.brightnessCount().getNaiveBrightness(),
    marker='',label='Naive'
)
plt.xlabel('Brightness')
plt.ylabel('Objects brighter')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.show()


