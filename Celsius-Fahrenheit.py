# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 07:27:53 2023

@author: Andres Florez
"""

# Programa que convierte ºC a ºF
C = float(input("Entra temperatura en ºC: "))
if C < -273.15:
    print('Temperatura en ºC irreal por debajo del 0 absoluto!!')
else:
    F = 1.8*C + 32
print(C, 'ºC equivale a', F, 'ºF')