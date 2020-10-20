# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 00:26:14 2020

@author: elohe
"""
class ClasesNum:
    def __init__(self, val_str_clase):
        self.val_int_clase = 8
        
        if (val_str_clase == 'BE'):
            self.val_int_clase = 1
            
        if (val_str_clase == 'FA'):
            self.val_int_clase = 2
        
        if (val_str_clase == 'PI'):
            self.val_int_clase = 3
        
        if (val_str_clase == 'RU'):
            self.val_int_clase = 4
        
        if (val_str_clase == 'SD'):
            self.val_int_clase = 5
        
        if (val_str_clase == 'SU'):
            self.val_int_clase = 6
        
        if (val_str_clase == 'WA'):
            self.val_int_clase = 7
            
