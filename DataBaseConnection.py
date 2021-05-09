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
                password = '',
                db = 'wifialr'
                )
        
        self.cursor = self.connection.cursor()
        print("¡Conexión establecida exitosamente!")
    
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
            
    def select_alertas(self, val_alerta):
        sql = "SELECT ALERTA FROM alertas WHERE FK_MOVIMIENTO = '%s'" % val_alerta 
    
        try:
            self.cursor.execute(sql)
            alerta = self.cursor.fetchone()
            return "ALERTA: UNA PERSONA " + alerta[0]
            
        except Exception as e:
            raise
            
    def select_movimientos(self, val_movimiento):
        sql = "SELECT SIGNIFICADO FROM movimientos WHERE ID = '%s'" % val_movimiento 
    
        try:
            self.cursor.execute(sql)
            movimiento = self.cursor.fetchone()
            return movimiento[0]
            
        except Exception as e:
            raise
            
    def close(self):
        self.connection.close()
        


