
'''Se importan los módulos que se utilizaran'''
import numpy  # necesaria para hallar de manera más rápida las raices asociadas a la velocidad
from pylab import *

'''Definimos las constantes'''
G = 6.6738E-11  # N*m^2/kg^2  La constante de gravitación universal
M_s = 1.9891E30  # kg  Masa del sol

'''El usuario ahora digita los datos(las variables)'''
l_1 = float(input('Ingrese la distancia del perihelio (m):\n'))
v_1 = float(input('Ingrese la velocidad del perihelio (m/s):\n'))

# Vamos a dejar unos input de prueba
# Tierra
l_1T = 1.4710E11  # m
v_1T = 3.0287E4  # m s^{-1}}
# Cometa Halley
l_1H = 8.7830E10  # m
v_1H = 5.4529E4  # m s^{-1}}

'''Ahora, definimos las operaciones necesarias para los cálculos, esto es:'''
# valor asociado a la conservación de energia en el perihelio
E_1 = v_1 ** 2 - 2 * G * (M_s / l_1)
# la conservación de energía en el perihelio de la Tierra
E_1T = v_1T ** 2 - 2 * G * (M_s / l_1T)
# la conservación de energía en el perihelio del cometa Halley
E_1H = v_1H ** 2 - 2 * G * (M_s / l_1H)
'''La ecuación cuadrática esta descrita por:
v**2-((2*G*M_s)/l_1*v_1)*v_2-E_1=0
donde relacionando con 
a_o*x**2+a_1*x+a_2=0 se tiene:
'''
a_0 = 1
a_1 = -(2 * G * M_s) / (l_1 * v_1)
a_2 = -E_1
# Definimos los coeficientes para los inputs de prueba
# Tierra
Ta_0 = 1
Ta_1 = -(2 * G * M_s) / (l_1T * v_1T)
Ta_2 = -E_1T
# Cometa Halley
Ha_0 = 1
Ha_1 = -(2 * G * M_s) / (l_1H * v_1H)
Ha_2 = -E_1H

'''Esta puede representarse en su forma LI como un vector dado por los coeficientes'''
pol_v_2 = [a_0, a_1, a_2]
# de igual forma para la Tierra
pol_v_2T = [Ta_0, Ta_1, Ta_2]
# polinomio Cometa Halley
pol_v_2H = [Ha_0, Ha_1, Ha_2]
''' Para hallar las raices del polinomio, utilizamos la función roots del
paquete numpy, y las guardamos en una variable raices'''
raices = numpy.roots(pol_v_2)
Traices = numpy.roots(pol_v_2T)  # raices velocidad afelio Tierra
Hraices = numpy.roots(pol_v_2H)  # raices velocidad afelio cometa Halley
'''
y ahora, debido a que el valor tomado es la raiz que sea distinta al valor ingresado
en la velocidad, ya que la velocidad del afelio es distinta a la del perihelio para 
conservar el momentum angular (2da ley de keppler), así, creamos un ciclo for 
para el arreglo de raices obtenidas y condicionamos para imprimir la raiz que sea
diferente al valor de velocidad ingresado(perihelio), tal qué
'''
for i in raices:
    if (round(i, 2) != v_1):
        v_2 = i
        print(f'\nLa velocidad del afelio es:\nv_2={v_2:0.4f} m/s')
        print(f'   ={v_2 * 3.6:4.4f} km/h\n')
    elif (e == 1):
        x = i
        print(f'\nLa velocidad del afelio es:\nv_2={x:4.4f} m/s')
        print(f'   ={x * 3.6:4.4f} km/h\n')
# Loop For para la Tierra
for j in Traices:
    if (round(j, 2) != v_1T):
        v_2T = j
# Loop For para cometa Halley
for k in Hraices:
    if (round(k, 2) != v_1H):
        v_2H = k

''' Luego, la longitud del afelio esta definida como: '''
l_2 = (v_1 * l_1) / v_2

# Afelio Tierra
l_2T = (v_1T * l_1T) / v_2T
# Afelio Halley
l_2H = (v_1H * l_1H) / v_2H

print(f'La distancia del afelio es:\nl_2={l_2 * 1000} km\n   ={l_2} m')
'''Definimos ahora las relaciones de la orbita:'''
a = 0.5 * (l_1 + l_2)  # semieje mayor
aT = 0.5 * (l_1T + l_2T)  # semieje mayor Tierra
aH = 0.5 * (l_1H + l_2H)  # semieje mayor Halley

b = numpy.sqrt(l_1 * l_2)  # semieje menor
bT = numpy.sqrt(l_1T * l_2T)  # semieje menor Tierra
bH = numpy.sqrt(l_1H * l_2H)  # semieje menor Halley

T = (2 * numpy.pi * a * b) / (l_1 * v_1)  # periodo orbital
Tierra = (2 * numpy.pi * aT * bT) / (l_1T * v_1T)  # periodo orbital Tierra
Halley = (2 * numpy.pi * aH * bH) / (l_1H * v_1H)  # periodo orbital Halley

e = (l_2 - l_1) / (l_2 + l_1)  # excentricidad
eT = (l_2T - l_1T) / (l_2T + l_1T)  # excentricidad Tierra
eH = (l_2H - l_1H) / (l_2H + l_1H)  # excentricidad Halley

print(f'\nPeriodo orbital: {T * 3.171e-8} años\n')
print(f'\nExcentricidad: {e}\n')

''' Ahora vamos a graficar lo encontrado para Tierra, el cometa Halley y el dato
digitado por el usuario
'''
# Definimos la cantidad de puntos a evaluar en la gráfica en un intervalo de x
x = numpy.linspace(int(-2 * l_2H), int(2 * l_2H), 100000)
# Se define ahora la función a graficar, esto es teniendo en cuenta
# (x/a)**2+(y/b)**2=1
ypos = b * (1 - (x / a) ** 2) ** (1 / 2)
yneg = -b * (1 - (x / a) ** 2) ** (1 / 2)

# Tambien para la Tierra
#FOCOS DE LA TIERRA
cH=(aH**2-bH**2)**(1/2)
cT=(aT**2-bT**2)**(1/2)
yposT = bT * (1 - ((x-cT) / aT) ** 2) ** (1 / 2)
ynegT = -bT * (1 - ((x-cT) / aT) ** 2) ** (1 / 2)

# Cometa Halley
'''CENTRADA EN EL FOCO(SOL)'''
yposH = bH * (1 - ((x-cH) / aH) ** 2) ** (1 / 2)
ynegH = -bH * (1 - ((x-cH) / aH) ** 2) ** (1 / 2)
'''CENTRADA EN EL ORIGEN DE LA ELIPSE'''
yposHH = bH * (1 - ((x) / aH) ** 2) ** (1 / 2)
ynegHH = -bH * (1 - ((x) / aH) ** 2) ** (1 / 2)
'''Graficamos los coeficientes'''
# PLOT DEL INPUT DE USUARIO
#plot(x*1000, ypos*1000, ls='--', color='royalblue',label='Datos ingresados', lw=2)
#plot(x*1000, yneg*1000, ls='--', color='royalblue', lw=2)
# PLOT PARA LA TIERRA
subplot(221)
resT = '$e=$' + str(round(eT, 3)) + '\n$T=$' + str(round(Tierra * 3.171e-8, 3)) + 'años'
plot(x * 1000/1e14, yposT * 1000/1e14, color='red', label=resT, lw=2)
plot(x * 1000/1e14, ynegT * 1000/1e14, color='red', lw=2)
grid(alpha=0.4, ls='--', color='grey')
xmax, xmin, ymin, ymax = axis()
xlim((xmax, xmin))
ylim(ymin, ymax)
title('Orbita Tierra')

# PLOT PARA COMETA HALLEY
subplot(222)
resH = '$e=$' + str(round(eH, 3)) + '\n$T=$' + str(round(Halley * 3.171e-8, 3)) + 'años'
plot(x * 1000/1e15, yposHH * 1000/1e14, ls='--', color='magenta', label=resH, lw=2)
plot(x * 1000/1e15, ynegHH * 1000/1e14, ls='--', color='magenta', lw=2)
grid(alpha=0.4, ls='--', color='grey')
xmax, xmin, ymin, ymax = axis()
xlim((xmax, xmin))
ylim(ymin, ymax)
title('Orbita Halley')

subplot(212)
plot(x * 1000/1e15, yposH * 1000/1e14, ls='--', color='magenta', label=resH, lw=2)
plot(x * 1000/1e15, ynegH * 1000/1e14, ls='--', color='magenta', lw=2)
plot(x * 1000/1e15, yposT * 1000/1e14, color='red', label=resT, lw=2)
plot(x * 1000/1e15, ynegT * 1000/1e14, color='red', lw=2)
grid(alpha=0.4, ls='--', color='grey')
xlabel(r'$x$(km)', fontsize=12)
ylabel(r'$y$(km)', fontsize=12)


xmax, xmin, ymin, ymax = axis()
xlim((xmax, xmin))
ylim(ymin, ymax)


legend(loc='center',  fontsize=12, ncol=3)
show()

