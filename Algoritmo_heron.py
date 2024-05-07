# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 09:35:31 2023

@author: Andres Florez
"""

#Algoritmo de Herón, cálculo de la raíz cuadrada

print('Cálculo de la raíz cuadrada de un número (algoritmo de Herón)')
x = float(input('Entra un número para hallar su raíz cuadrada: '))
g = x/2  # El valor inicial g (guess), g0, puede ser x/2
epsilon = 1e-10 # Tolerancia de la aproximación a la raíz cuadrada
dif = g*g - x # Diferencia entra el valor de raíz g0 propuesto y el real

while abs(dif) > epsilon:
    g = (g+x/g) /2
    dif = g*g - x
    print('Raiz de ',x, 'aprox:', g) # print usado para rastrear la búsqueda
print('la raíz cuadrada aproximada de ',x, 'es' ,g)