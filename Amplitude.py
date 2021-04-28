# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 02:47:40 2021
Script para la extracción de la Amplitud de Señales CSI.

@author: elohe
"""

from CSIKit.util.matlab import db

import numpy as np

def get_CSI_Frames(csi_data: 'CSIData'):
    
    frames = csi_data.frames

    csi_shape = frames[0].csi_matrix.shape

    no_frames = len(frames)
    no_subcarriers = csi_shape[0]

    csi_1 = np.zeros((no_subcarriers, no_frames))
    csi_2 = np.zeros((no_subcarriers, no_frames))
    csi_3 = np.zeros((no_subcarriers, no_frames))

    for x in range(no_frames): # 20,000 frames aprox
        entry = frames[x].csi_matrix
        for y in range(no_subcarriers): # 30 subcarriers
            csi_1[y][x] = db(abs(entry[y][0]))
            csi_2[y][x] = db(abs(entry[y][1]))
            csi_3[y][x] = db(abs(entry[y][2]))
    return (csi_1, csi_2, csi_3)