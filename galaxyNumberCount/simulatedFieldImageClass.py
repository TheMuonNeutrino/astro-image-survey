import numpy as np
from .fieldImageClass import FieldImage
import multiprocessing

import numpy as np
rng = np.random.default_rng()

class SimulatedObject():
    def __init__(self,minMagnitude,maxMagnitude,minStd,maxStd):
        self.minMagnitude = minMagnitude
        self.maxMagnitude = maxMagnitude
        self.minStd = minStd
        self.maxStd = maxStd

    def generateLoc(self,shape):
        self.x = rng.uniform(high=shape[0])
        self.y = rng.uniform(high=shape[1])

    def generateMagnitude(self):
        c = -0.6 * self.maxMagnitude
        d = - np.exp(0.6 * self.minMagnitude + c)
        p_cum = rng.uniform()
        m = (np.log10(p_cum - d)-c)/0.6
        self.magnitude = m

    def generateShape(self):
        self.std_1 = rng.uniform(self.minStd,self.maxStd)
        self.std_2 = rng.uniform(self.minStd,self.maxStd)
        self.angle = rng.uniform(0,np.pi)

    def place(self):
        Xgrid = self.read_Xgrid
        Ygrid = self.read_Ygrid
        a = np.cos(self.angle)**2 / (2 * self.std_1**2) + np.sin(self.angle)**2 / (2 * self.std_2**2)
        b = -np.sin(2 * self.angle) / (4 * self.std_1**2) + np.sin(2 * self.angle) / (4 * self.std_2**2)
        c = np.sin(self.angle)**2 / (2 * self.std_1**2) + np.cos(self.angle)**2 / (2 * self.std_2**2)

        return np.power(self.magnitude/2.5,10) * np.exp(
            -(a * (Xgrid - self.x)**2 + 2 * b * (Xgrid - self.x) * (Ygrid - self.y) + c * (Ygrid - self.y)**2)
        )

def placeFromObject(o):
    return o.place()

class SimulatedFieldImage(FieldImage):
    def __init__(self,
        shape = (2000,2000),
        backgroundParams={'loc':3420,'scale':12},
        nObjects=50,
        objectParams={
            'minMagnitude': 10,
            'maxMagnitude': 17.5,
            'minStd': 3,
            'maxStd': 6,
        }
    ):
        super().__init__()
        self.sim_shape = shape
        self.sim_backgroundParams = backgroundParams
        self.sim_nObjects = nObjects
        self.sim_objectParms = objectParams
        self.sim_objects = []

    def simulate(self):
        self.genObjects()
        self.placeObjects()
        self.placeBackground()
    
    def genObjects(self):
        X = np.arange(0,self.sim_shape[0])
        Y = np.arange(0,self.sim_shape[1])
        Xgrid, Ygrid = np.meshgrid(X,Y)

        for i in range(self.sim_nObjects):
            o = SimulatedObject(**self.sim_objectParms)
            o.generateMagnitude()
            o.generateShape()
            o.generateLoc(self.sim_shape)
            o.read_Xgrid = Xgrid
            o.read_Ygrid = Ygrid
            
            self.sim_objects.append(o)

    def placeObjects(self):
        U_total = np.zeros(self.sim_shape)
        with multiprocessing.Pool(5) as p:
            for i, U in enumerate(p.imap(placeFromObject,self.sim_objects)):
                U_total += U
                print(f'Generated {i+1} / {self.sim_nObjects} objects   ',end='\r')
                
        self.image = U_total

    def placeBackground(self):
        self.image += rng.normal(**self.sim_backgroundParams,size=self.image.shape)