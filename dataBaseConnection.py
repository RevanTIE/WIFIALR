# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:36:45 2020

@author: elohe
"""

import pymysql

class DataBase:
    def __init__(self):
        self.connection = pymysql.connect(
                host = 'localhost', #ip
                user = 'root',
                password = 'Wifialr3.',
                db = 'wifialr'
                )
        
        self.cursor = self.connection.cursor()
        print("Conexión establecida exitosamente!")
    
    #Función de ejemplo
    def select_movimiento(self):
        sql = "SELECT * FROM movimientos"
    
        try:
            self.cursor.execute(sql)
            ##self.connection.commit()
            movimiento = self.cursor.fetchall()
            
            for mov in movimiento:
                print("Id: ", mov[0], "Movimiento: ", mov[1])
            
        except Exception as e:
            raise
            
    def close(self):
        self.connection.close()
        


