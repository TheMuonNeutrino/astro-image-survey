import numpy as np
import matplotlib.pyplot as plt
import warnings
from os import path

from .core import pad
from .astronomicalObjectClass import AstronomicalObject

def saveObjectPlot(object: AstronomicalObject,i,folderpath):
    warnings.filterwarnings('ignore')
    fig = plt.figure()
    object.plotPixelsAndCentres()
    plt.savefig(path.join(folderpath,f'{pad(i,4)}_{pad(object.id,4)}.png'))
    plt.close()
    warnings.resetwarnings()

def saveTwinPlot(object: AstronomicalObject,i,folderpath):
    warnings.filterwarnings('ignore')
    fig, axs = plt.subplots(2,min(4,len(object.splitDaughterIds)))
    plt.sca(axs[0][0])
    object.plotPixelsAndCentres()
    for j,id in enumerate(object.splitDaughterIds):
        if j < 4:
            daughter: AstronomicalObject = object.parentImageField.objects[id]
            plt.sca(axs[1][j])
            daughter.plotPixelsAndCentres()
            axs[1][j].get_legend().remove()
    plt.tight_layout()
    plt.savefig(path.join(folderpath,f'{pad(i,4)}_{pad(object.id,4)}.png'))
    plt.close()
    warnings.resetwarnings()