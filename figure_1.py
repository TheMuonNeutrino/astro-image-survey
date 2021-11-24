
from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
import pickle
from galaxyNumberCount.astronomicalObjectClass import AstronomicalObject
from galaxyNumberCount import core, plottools
from galaxyNumberCount.fieldImageClass import FieldImage
from galaxyNumberCount.simulatedFieldImageClass import SimulatedObject

from galaxyNumberCount.utilities import bcolors, clearFolder, printC

rng = np.random.default_rng(3002)

ROOT_PATH = path.dirname(__file__)
CACHE_PATH = path.join(ROOT_PATH,'FieldImageCache.pickle')

SMALL_SIZE = 8+4
MEDIUM_SIZE = 10+4
BIGGER_SIZE = 12+4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=MEDIUM_SIZE)


with open(CACHE_PATH,'rb') as file:
    img: FieldImage = pickle.load(file)

def genOutlineSegments(mapimg):
    # https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
        # a vertical line segment is needed, when the pixels next to each other horizontally
    #   belong to diffferent groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates 
    ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])

    # the same is repeated for horizontal segments
    hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add Nones in between segments
    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan,np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))

    # now we transform the list into a numpy array of Nx2 shape
    segments = np.array(l)

    # now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points
    segments[:,0] = segments[:,0] - 0.5
    segments[:,1] = segments[:,1] - 0.5

    return segments[:,0], segments[:,1]


objectA = img.objects[257]
pixels = objectA.croppedPixel
map = objectA.croppedMask

# fig = plt.figure()
# ax = plt.subplot2grid((1,3),(0,1),1,2)
# im = plt.imshow(pixels,norm=matplotlib.colors.LogNorm())
# cbar = fig.colorbar(im)
# cbar.ax.set_ylabel('Pixel intensity')
# cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
# cbar.ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
# plt.axis('off')

# ax = plt.subplot2grid((1,3),(0,0),1,1)
# pixels2 = np.array([[0,0.3,0],[0.3,1,0.3],[0,0.3,0]])
# plt.imshow(pixels2,cmap='Greys')
# plt.axis('off')

# plt.tight_layout()
# plt.show()



if False:
    objectB = img.objects[484]
    pixels = objectB.croppedPixel
    pixels = np.hstack([pixels,np.zeros((pixels.shape[0],5))])
    pixels[5,4] = 3611
    pixels[2,7] = 3612
    simObject = SimulatedObject(0,0,0,0)
    simObject.x = 4
    simObject.y = 15
    simObject.magnitude = -4.215
    simObject.std_1 = 2
    simObject.std_2 = 2
    simObject.angle = 30 / 180 * np.pi
    X = np.arange(0,pixels.shape[0])
    Y = np.arange(0,pixels.shape[1])
    Xgrid, Ygrid = np.meshgrid(X,Y)
    simObject.read_Xgrid = Xgrid
    simObject.read_Ygrid = Ygrid
    pixelsAdded = simObject.place()
    pixels[pixels == 0] = rng.normal(loc=3420,scale=12,size=pixels.shape)[pixels == 0]
    pixels = pixels + pixelsAdded.astype(np.int16).T
    pixels[pixels < 3480] = 0
    pixels = pixels[:,:-2]
    Xgrid = Xgrid[:-2]

    fig = plt.figure()
    ax = plt.subplot2grid((3,3),(0,0),2,3)
    myNorm = matplotlib.colors.LogNorm()
    im = plt.imshow(pixels, norm = myNorm,zorder=-20)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Pixel intensity',labelpad=10)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    cbar.ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    cbar.ax.axhline(3612,color='r')
    plt.axis('off')
    plt.plot(*genOutlineSegments(
        (pixels > 3612) & (Xgrid > 5).T
    ),color=(0.8,0,0,.7), linewidth=3,label='Region A')
    plt.plot(*genOutlineSegments(
        (pixels > 3612) & (Xgrid < 5).T
    ),color='#ff9900ee', linewidth=3, label='Region B')
    plt.plot(*genOutlineSegments(
        (pixels > 3611) & (pixels < 3612.5)
    ),color=(0,1,0,.7), linewidth=3, ls=(0,(1,1)), zorder=10, label='Pixels to add')
    plt.legend(loc='lower left',bbox_to_anchor=(-0.2, 0))

    objectB.bbox = list(objectB.bbox)
    objectB.bbox[1] += 3
    img.image[core.indexFromBbox(objectB.bbox)] = pixels
    objectB.croppedPixel = pixels
    objectB.attemptTwinSplit(False)

    for i, id in enumerate(objectB.splitDaughterIds):
        daughter = img.objects[id]
        ax_i = plt.subplot2grid((3,3),(2,i),1,1)
        pixel = daughter.croppedPixel
        if i != 0:
            pixel = np.vstack([np.zeros(pixel.shape[1]),pixel,np.zeros(pixel.shape[1])])
        im = plt.imshow(pixel, norm = myNorm,zorder=-20)
        if i == 1:
            plt.title('Output galaxies')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

objectD = img.objects[215] 
objectD = img.objects[180]

rFunc = lambda w, h: int( np.max([3,np.max([w,h])*0.8]) // 1 )
r = rFunc(*objectD.shape)
sliceIndex, placementMatrix, aperture = objectD._getCroppedCircularAperture(r,r)
includeMask = ~objectD._maskOtherObjectsAndEdge(r,0)
pixelsInAperture = img.image[sliceIndex]
pixelsInAperture[~includeMask] = 0
allPixelsInAperture = pixelsInAperture.copy()
pixelsInAperture[img.globalObjectMask[sliceIndex] & (pixelsInAperture != 0)] = 0

fig = plt.figure()
ax1 = plt.subplot2grid((2,3),(0,0),1,1)
#myNorm2 = matplotlib.colors.LogNorm()

midpointShift = 0.1
colors0 = np.array([[1,1,1,1],])
cmap = plt.cm.viridis
colors1 = cmap(np.linspace(0., 0.6, 127))
colors2 = cmap(np.linspace(0.6, 1., 128))

colors = np.vstack((colors0,colors1, colors2))
mymap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', colors)

CMAP = mymap
myNorm2 = matplotlib.colors.TwoSlopeNorm(
    img.galaxy_significance_threshold - 0.5 * (img.galaxy_significance_threshold - img.backgroundMean),
    np.min(pixelsInAperture[pixelsInAperture > 0])-1,
    np.max(objectD.croppedPixel)
)
im = plt.imshow(objectD.croppedPixel,norm = myNorm2,cmap=CMAP)
plt.axis('off')

ax2 = plt.subplot2grid((2,3),(0,1),2,2)
#myNorm3 = matplotlib.colors.LogNorm()
im2 = plt.imshow(allPixelsInAperture,norm=myNorm2,cmap=CMAP)
cbar = fig.colorbar(im2,spacing='proportional')
cbar.ax.set_ylabel('Pixel intensity',labelpad=10)
plt.axis('off')

plt.tight_layout()
cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
cbar.ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
cbar.set_ticks([3400,3430,3460,3500,3750,4000,4250])

X = np.arange(0,pixelsInAperture.shape[0])
Y = np.arange(0,pixelsInAperture.shape[1])
Xgrid, Ygrid = np.meshgrid(X,Y)

plt.plot(*genOutlineSegments(
    (pixelsInAperture > 3460) & (Xgrid < 19).T
),color=(0.8,0,0,.7), linewidth=2,label='Border pixels of object')
plt.plot(*genOutlineSegments(
    (pixelsInAperture > 3460) & (Xgrid >= 19).T
),color='#ff9900ee', linewidth=2, label='Pixels in other object')
plt.legend(loc='lower left',bbox_to_anchor=(-0.8, 0.0))

plt.show()