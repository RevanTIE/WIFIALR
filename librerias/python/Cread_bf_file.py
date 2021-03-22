# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:09:48 2021

@author: elohe
"""

class ClassReadBfFile:
    def __init__(self, filename):
        import math
        import pandas as pd
        import sys
        
        # Aquí se definirán las condiciones de error
        try:
            f = open(filename, "rb")
            """
            if (f<0):
                raise Exception("No se puede abrir el archivo", filename)
            """
            #Variable definition
            len= f.tell()
            
            #listofzeros = [0] * math.ceil(len/95)
            #self.ret =  pd.MultiIndex.from_arrays([listofzeros]) #Holds the return values - 1x1 CSI is 95 bytes big, so this should be upper bound
            
            self.ret = [{}] * math.ceil(len/95)
            cur = 0  #Current offset into file                    
            count = -1  #Number of records output                
            broken_perm = 0 #Flag marking whether we've encountered a broken CSI yet               
            triangle = (1, 3, 6)  #What perm should sum to for 1,2,3 antennas           
            
            #Se procesan todas las entradas del archivo
            while cur < (len - 3):
               # Leer tamaño y código
                field_len = f.read(1, 'uint16', 0, 'ieee-be')
                code = f.read(1)
                cur = cur+3
                
                #If unhandled code, skip (seek over) the record and continue
                if (code == 187): # get beamforming or phy data
                    bytes = f.read(field_len-1, 'uint8=>uint8')
                    cur = cur + field_len - 1
                    if (len(bytes) != field_len-1):
                        f.close()
                        return
                    
                else: # skip all other info
                    f.seek(field_len - 1, 'cof')
                    cur = cur + field_len - 1
                    continue
                
                if (code == 187): #hex2dec('bb')) Beamforming matrix -- output a record
                    count = count + 1
                    self.ret[count] = read_bfee(bytes)
                    
                    perm = self.ret[count].perm
                    Nrx = self.ret[count].Nrx
                    if Nrx == 1: # No permuting needed for only 1 antenna
                        continue
                    
                    if sum(perm) != triangle(Nrx): # matrix does not contain default values
                        if broken_perm == 0:
                            broken_perm = 1
                            print('WARN ONCE: Found CSI {} with Nrx={} and invalid perm={}\n'.format(filename, Nrx, str(perm)))
                    else:
                        self.ret[count].csi[:,perm[1:Nrx],:] = self.ret[count].csi[:,1:Nrx,:]
            
            self.ret = self.ret[1:count]
            f.close()
            
        except:
            print("Se ha producido un error en 'read_bf_file'", sys.exc_info()[0])
            
        finally:
            f.close()
            return