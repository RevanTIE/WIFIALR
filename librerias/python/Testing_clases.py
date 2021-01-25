# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:15:05 2021

@author: elohe
"""
"""
from Cdbinv import ClassDbinv
import pandas as pd
import string
"""
from tkinter import Tk 
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
from tkinter import re #operaciones de matching de expresiones regulares
#import decimal

"""----------------------------
Test Cdbinv
"""
"""
valor_aleatorio = 2
calculo = ClassDbinv(valor_aleatorio).dbinv
-------------------------------
"""
"""
n = 6
#listofzeros = [{}] * n

listofzeros = [0] * n

ret =  pd.MultiIndex.from_arrays([listofzeros])
--------------------------------

PathName= "C:/Archivos de Programa/"
FileName_new = "Archivo"

Union = PathName + FileName_new + ".csv"
-------------------------------
   """ 
#x = -1

#if x < 0:
#  raise Exception("Sorry, no numbers below zero") 

 #import tkFileDialog

    #import re

    #ff = tkFileDialog.askopenfilenames()

    #files = re.findall('{(.*?)}', ff)

"""
import Tkinter,tkFileDialog

root = Tkinter.Tk()

files = tkFileDialog.askopenfilenames(parent=root,title='Choose a file')

#files = raw_input("which files do you want processed?")

files = root.tk.splitlist(files)

print ("list of filez =",files)

triangle = (1, 3, 6)
temp = [1, 2]
---------------------------------------
"""
#### Apertura de Dialogbox para selección de archivos

#Se abre una ventana de dialogo para solicitar el archivo csv
root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra
file_path = askopenfilenames(parent=root,title='Choose a file')

for i in range(len(file_path)):
    splitted = file_path[i].split("/")
    file_name = splitted[-1]
    folder_name = file_path[i].replace(file_name,'')  

#Para guardar el directorio donde se encontraba el archivo seleccionado
##match = re.search(r'/.*\..+', file_path) #matches name of file
##file_position = file_path.find(match.group()) #defines position of filename in file path

##save_path = file_path[0: file_position+1] #extracts the saving path.