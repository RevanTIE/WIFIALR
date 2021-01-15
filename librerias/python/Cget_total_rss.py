# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 23:36:02 2021

@author: elohe
"""

class ClassGetTotalRss:
    def __init__(self, csi_st):
        from Cdbinv import ClassDbinv
        import math
        ## Condición de error puede manejarse con try - catch
        self.rssi_mag = 0
        
        if (csi_st.rssi_a != 0):
            self.rssi_mag = self.rssi_mag + ClassDbinv(csi_st.rssi_a).dbinv
            
        if (csi_st.rssi_b != 0):
            self.rssi_mag = self.rssi_mag + ClassDbinv(csi_st.rssi_b).dbinv
        
        if (csi_st.rssi_c != 0):
            self.rssi_mag = self.rssi_mag + ClassDbinv(csi_st.rssi_c).dbinv

        self.ret = (10*math.log10(self.rssi_mag)) - 44 - csi_st.agc