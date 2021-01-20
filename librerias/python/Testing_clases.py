# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:15:05 2021

@author: elohe
"""

from Cdbinv import ClassDbinv
import pandas as pd

"""----------------------------
Test Cdbinv
"""
"""
valor_aleatorio = 2
calculo = ClassDbinv(valor_aleatorio).dbinv
-------------------------------
"""

n = 6
#listofzeros = [{}] * n

listofzeros = [0] * n
ret =  pd.MultiIndex.from_arrays([listofzeros])

#x = -1

#if x < 0:
#  raise Exception("Sorry, no numbers below zero") 