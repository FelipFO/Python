
'''Se importan los módulos que se utilizaran'''
from scipy import constants

'''para importar constantes'''
import numpy as np

'''Operaciones lineales'''
from pylab import *

'''modulo de matplotlib para graficar'''

'''Definimos las constantes'''
h = constants.hbar * (6.242e18 / 1.0)
m = 9.11e-31
E = 10.0
V0 = 9.0
'''Definimos una linea de puntos para el potencial'''
V = np.linspace(0, 10, 90)
'''Definimos cantidades iniciales de vectores de onda'''
k10 = ((2.0 * m * E) ** (1 / 2)) / h
k20 = ((2.0 * m * (E - V0)) ** (1 / 2)) / h
'''Definimos coeficientes de transmision y reflexion'''
T0 = (4.0 * k10 * k20) / ((k10 + k20) ** 2)
R0 = ((k10 - k20) / (k10 + k20)) ** 2
'''Imprimimos en pantalla los valores'''
print(T0)
print(R0)
print(T0 + R0)
'''Definimos los vectores de onda que dependan del potencial'''
k1 = ((2.0 * m * E) ** (1 / 2)) / h
k2 = ((2.0 * m * (E - V)) ** (1 / 2)) / h
T = (4.0 * k1 * k2) / ((k1 + k2) ** 2)
R = ((k1 - k2) / (k1 + k2)) ** 2
'''Graficamos los coeficientes'''
plt.rcParams["font.family"] = "serif"
plot(V, T, ls='--', color='royalblue', lw=3, label=r'$T = \dfrac{4\,k_1\,k_2}{(k_1+k_2)^2}$', dashes=(2, 1, 1, 1))
title('Probabilidad de transmision y reflexion', fontsize=20)
plot(V, R, ls='--', color='red', lw=3, label=r'$R = (\dfrac{k_1-k_2}{k_1+k_2})^2$', dashes=(2, 1, 1, 1))
axvline(9, color='green', label=r'$V_0$', lw=3)
xlabel("V (eV)", fontsize=16)
ylabel("P (T, R)", fontsize=16)
plot(V, T + R, color='magenta', lw=3, label='T + R')
legend(loc='center left', fontsize=16)
grid(alpha=0.4,ls='--',color='grey')
show()