import matplotlib.pyplot as plt
import numpy as np

x=float(input('Digite la distancia x(en años luz) desde el marco en reposo:\n'))
v=float(input('Digite la velocidad de la nave como una fracción de la velocidad de la luz:\n'))

'''Constantes'''
c=1
beta=v/c
gamma=(1 - beta**2)**(-1/2)
'''Operaciones'''
t_inercial=x/v
t_movil=t_inercial*gamma**(-1)

'''Salidas'''
print(f'El tiempo del observador inercial es:\n{t_inercial:0.2f} años')
print(f'El tiempo del observador móvil es:\n{t_movil:0.2f} años')

'''Graficar lás lineas de mundo'''
L = np.linspace(0,int(x+3),1000)
t_inercialgraph=L/v
t_movilgraph=t_inercialgraph*gamma**(-1)
plt.plot(L,t_inercialgraph,label='Marco inercial',color='blue')
plt.plot(L,t_movilgraph,label='Marco móvil',color='red')
plt.plot(L,L,ls='--',label='Cono de luz',color='lime')
plt.xlabel('$x$ (añosluz)')
plt.ylabel('$t$ (años)')
plt.legend(loc='upper left', fontsize=12)
plt.plot(x,t_inercial,marker='o',color='blue')
plt.text(x+0.1,t_inercial-0.5,(''+str(round(t_inercial,2))),color='blue')
plt.plot(x,t_movil,marker='o',color='red')
plt.text(x+0.1,t_movil+0.5,(''+str(round(t_movil,2))),color='red')
plt.grid(alpha=0.4,ls='--',color='grey')
xmin,xmax,ymin,ymax = plt.axis()
plt.xlim((xmin,xmax-1))
plt.ylim(ymin,ymax-1)
plt.yticks(np.arange(0,ymax,int(ymax/5)))
plt.xticks(np.arange(0,xmax))
plt.title('Lineas de mundo para $x=$'+str(x)+' añosluz, $v=$'+str(v)+'c')

plt.show()
