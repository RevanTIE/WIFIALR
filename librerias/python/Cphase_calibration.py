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
        difference = 0
        
        for i in range(1,30):
            temp = phasedata[i] - phasedata[i-1]
            
            if (abs(temp) > math.pi): 
                difference = self.difference + 1*np.sign[temp]
                
            self.calibrated_phase[i] = phasedata[i] - difference * 2 * math.pi
            
        k = (self.calibrated_phase[29] - self.calibrated_phase[0]) / (30 - 1)
        b = mean(self.calibrated_phase)
        
        for i in range(30):
            self.calibrated_phase2[i] = self.calibrated_phase[i] - k * i - b