# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:34:51 2021

@author: elohe
"""

class ClassGetScaledCsi:
    def __init__(self, csi_st):
        from Cdbinv import ClassDbinv
        from Cget_total_rss import ClassGetTotalRss
        import numpy as np
        
        csi = csi_st.csi
        csi_sq = csi * np.conj(csi)
        
        csi_pwr = np.sum(csi_sq[:]) ##Revisar si así se crea un vector de columna a partir de la matriz
        rssi_pwr = ClassDbinv(ClassGetTotalRss(csi_st).ret).dbinv
        scale = rssi_pwr / (csi_pwr / 30)
        
        if (csi_st.noise == -127):
            noise_db = -92
        else:
            noise_db = csi_st.noise
            
        thermal_noise_pwr = ClassDbinv(noise_db).dbinv
        
        quant_error_pwr = scale * (csi_st.Nrx * csi_st.Ntx)
        
        total_noise_pwr = thermal_noise_pwr + quant_error_pwr
        
        self.ret = csi * np.sqrt(scale / total_noise_pwr);
        if csi_st.Ntx == 2:
            self.ret = self.ret * np.sqrt(2)
        elif csi_st.Ntx == 3:
            self.ret = self.ret * np.sqrt(ClassDbinv(4.5).dbinv)
        
