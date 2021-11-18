import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from galaxyNumberCount.simulatedFieldImageClass import SimulatedFieldImage
import pickle
import scipy.stats
from os import path
from galaxyNumberCount.utilities import printC, bcolors

from main import ROOT_PATH

ROOT_PATH = ''
PRE_CACHE_PATH = path.join(ROOT_PATH,'SimFieldImageCache_Pre.pickle')
POST_CACHE_PATH = path.join(ROOT_PATH,'SimFieldImageCache_Post.pickle')

USE_PRE_CACHE = False

if __name__ == '__main__':

    if not USE_PRE_CACHE:
        simField = SimulatedFieldImage(nObjects=2000,shape=(2000,2000))
        simField.simulate()

        plt.imshow(simField.image)
        plt.show()

        with open(PRE_CACHE_PATH,'wb') as file:
            pickle.dump(simField,file)
    if USE_PRE_CACHE:
        with open(PRE_CACHE_PATH,'rb') as file:
            simField = pickle.load(file)

    simField.backgroundFitCoeffs = [1,simField.sim_backgroundParams['loc'],simField.sim_backgroundParams['scale'],0,0,1]
    simField.backgroundMean = simField.sim_backgroundParams['loc']
    simField.backgroundStd = simField.sim_backgroundParams['scale']
    simField.galaxy_significance_threshold = simField.backgroundMean + 4 * simField.backgroundStd
    simField.header = {'MAGZPT': 23.5}
    simField.printBackgroundInfo()
    simField.identifyObjects(
        simField.galaxy_significance_threshold + 0 * simField.backgroundStd,
        simField.galaxy_significance_threshold - 2 * simField.backgroundStd
    )
    simField.seperateTwins()

    with open(POST_CACHE_PATH, 'wb') as file:
        pickle.dump(simField,file)

    extractionFunc = simField.magnitudeCountBinned
    rFunc = lambda w, h: int( np.max([6,np.max([w,h])*0.8]) // 1 )
    numberCounts = {
        'Naive | Subtracted background': extractionFunc().getBrightnessWithoutBackground(),
        'Naive | Local background': extractionFunc().getBrightnessWithoutLocalBackground(
            rBackground=30,dilateObjectMaskBackground=6,minimumPixels=50
        ),
        'Aperture | Subtracted background': extractionFunc().getCircularApertureBrightness(
            rFunc
        ),
        'Aperture | Local background': extractionFunc().getCircularApertureBrightness(
            rFunc,'local',rBackground=30,dilateObjectMaskBackground=6
        )
    }

    for key, (xMagnitude, nBright) in numberCounts.items():
        nBrighter = np.cumsum(nBright)
        nBrighter_err = np.sqrt(nBright)
        indicies = (xMagnitude >= 12) & (xMagnitude < 16)
        indicies = nBrighter != 0
        xMagnitudeFit = xMagnitude[indicies]
        nBrighterFit = nBrighter[indicies]
        result = scipy.stats.linregress(xMagnitudeFit, np.log(nBrighterFit))
        intercept_std_err = result.intercept_stderr
        slope, intercept, r_value, p_value, std_err = result
        r_squared = r_value**2

        printC(bcolors.OKGREEN,f'{key} | r^2: {r_squared}')
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