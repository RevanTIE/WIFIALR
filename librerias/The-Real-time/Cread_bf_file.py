# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:09:48 2021

@author: elohe
"""
import read_bfee
import numpy as np
import math
import pandas as pd
import sys
import struct

SIZE_STRUCT = struct.Struct(">H").unpack
CODE_STRUCT = struct.Struct("B").unpack

def read_bf_file(filename):
    # Aquí se definirán las condiciones de error
    #try:
    fd = open(filename, "rb").read()
    """
    if (f<0):
        raise Exception("No se puede abrir el archivo", filename)
    """
    #Variable definition
    ##l_en= fd.tell()
    l_en = len(fd)
    flo = np.ceil(l_en/95)
    ret = ([{}]) * int(flo)

    #ret = np.zeros((np.ceil(l_en/95), 1))
    
    cur = 0  #Current offset into file
    count = -1  #Number of records output
    broken_perm = 0 #Flag marking whether we've encountered a broken CSI yet
    triangle = [1,3,6]  #What perm should sum to for 1,2,3 antennas
    ##p = np.zeros((30, 30), dtype='double', order='C')

    #Se procesan todas las entradas del archivo
    while cur < (l_en - 3):
        # Leer tamaño y código
        """
        fl = fd.read(2)
        field_len = int.from_bytes(fl, 'big')
        ##if field_len == 0:
          ##  break

        co = fd.read(1)
        #if not isinstance(co, int):
        code = int.from_bytes(co, 'big')
        #else:
         #   code = co
         """
        field_len = SIZE_STRUCT(fd[cur:cur + 2])[0]
        code = CODE_STRUCT(fd[cur + 2:cur + 3])[0]

        cur += 3

        #If unhandled code, skip (seek over) the record and continue
        if code == 187: # get beamforming or phy data
            bytes = fd[cur:cur+field_len-1]
            cur = cur + field_len - 1
            if (len(bytes) != field_len-1):
                #fd.close()
                exit()

        else: # skip all other info
            fd.seek(field_len - 1, 'cof')
            cur = cur + field_len - 1
            continue

        if code == 187: #hex2dec('bb')) Beamforming matrix -- output a record
            count = count + 1
            ret[count] = read_bfee.read_bfee(bytes)
            if not ret:
                print('Error: malformed packet')
                exit()

            perm = ret[count][9]
            Nrx = ret[count][2]
            if Nrx == 1: # No permuting needed for only 1 antenna
                continue
            
            if sum(perm) != triangle[Nrx-1]: # matrix does not contain default values
                if broken_perm == 0:
                    broken_perm = 1
                    print('WARN ONCE: Found CSI with Nrx=', Nrx, ' and invalid perm=\n')
                    ##print('WARN ONCE: Found CSI {} with Nrx={} and invalid perm={}\n'.format(filename, Nrx, str(perm)))
                else:
                    ret[count][11][:,perm[0:Nrx],:] = ret[count][11][:,0:Nrx,:]

    ##Aqui termina while
    ret = ret[1:count]
    #fd.close()
    return ret
    """
    except:
        print("Se ha producido un error en 'read_bf_file'", sys.exc_info()[0])

    finally:
        fd.close()
        return
    """