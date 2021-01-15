# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 01:49:17 2021

@author: elohe
"""

class ClassPhaseCalibration:
    def __init__(self, phasedata):
        from statistics import mean
        import math
        import numpy as np
        
        self.calibrated_phase[0] = phasedata[0]
        self.difference = 0
        
        for i in range(1,30):
            self.temp = phasedata[i] - phasedata[i-1]
            
            if (abs(self.temp) > math.pi): 
                self.difference = self.difference + 1*np.sign[self.temp]
                
            self.calibrated_phase[i] = phasedata[i] - self.difference * 2 * math.pi
            
        self.k = (self.calibrated_phase[29] - self.calibrated_phase[0]) / (30 - 1)
        self.b = mean(self.calibrated_phase)
        
        for i in range(30):
            self.calibrated_phase2[i] = self.calibrated_phase[i] - self.k * i - self.b;