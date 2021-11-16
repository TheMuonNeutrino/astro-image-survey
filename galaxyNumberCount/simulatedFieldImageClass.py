import numpy as np
from fieldImageClass import FieldImage

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
        self.angle = rng.uniform(0,2*np.pi)

    def place(Xgrid,Ygrid,Ugrid):
        pass        
    

class simulatedFieldImage(FieldImage):
    def __init__(self,
        shape = (2000,2000),
        backgroundParams={'loc':3420,'scale':12},
        nObjects=700,
        objectParams={
            'minMagnitude': 10,
            'maxMagnitude': 20,
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
        for i in range(self.sim_nObjects):
            pass
