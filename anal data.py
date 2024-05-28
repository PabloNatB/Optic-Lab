# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:04:08 2023

@author: pablo
"""
#importamos librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
df1=pd.read_excel('D:/AA_Trabajos_Escuela/7mo sem/Óptica/Practica 4/Práctica 4 - Polarización.xlsx',sheet_name="Parte1.2")
x1=np.array(df1['Ángulo (grados)'])
y1=np.array(df1['Amplitud (mili-volts)'])
plt.figure()
plt.title('Datos en bruto')
plt.xlabel('Ángulo (grados)')
plt.ylabel('Amplitud (mili-volts)')
plt.plot(x1,y1,label='Experimento 1')

df2=pd.read_excel('D:/AA_Trabajos_Escuela/7mo sem/Óptica/Practica 4/Práctica 4 - Polarización.xlsx',sheet_name="Parte2")
x2=np.array(df2['Ángulo (grados)'])
y2=np.array(df2['Amplitud (volts)'])
plt.plot(x2,y2,label='Experimento 2')
plt.legend()
#plt.savefig('D:/AA_Trabajos_Escuela/7mo sem/Óptica/Practica 4/datosbruto.png')
plt.show()

def funcion(x,A,B):
    return (A**2*np.cos(x)**2+B**2*np.sin(x)**2)/(np.sqrt(A**2+B**2))

x1=x1*(np.pi/180)

params, cov = curve_fit(funcion, x1, y1)

A,B=params[0],params[1]

xn=np.linspace(0,2*np.pi,100)
yn=[funcion(xn[i],A,B) for i in range(100)]

plt.figure()
plt.scatter(x1,y1,label='Experimental',color='green',marker='x')
plt.plot(xn,yn,label='Ajuste')
#plt.plot(x,y,label='Ajuste. A='+str(round(A,2))+' B='+str(round(B,2)))
plt.xlabel('Ángulo en radianes')
plt.ylabel('Intensidad')
plt.legend()
plt.show()
print(A)
print(B)

def f2(x,A,B):
    return ((A**2)*np.cos(x)**4)/(np.sqrt(A**2+B**2))


x2=x2*(np.pi/180)

params, cov = curve_fit(funcion, x2, y2)

A,B=params[0],params[1]

x=np.linspace(0,2*np.pi,100)
y=[funcion(x[i],A,B) for i in range(100)]

plt.figure()
plt.scatter(x2,y2,label='Experimental 2',color='green',marker='x')
plt.plot(x,y,label='Ajuste 2')
#plt.plot(x,y,label='Ajuste. A='+str(round(A,2))+' B='+str(round(B,2)))
plt.xlabel('Ángulo en radianes')
plt.ylabel('Intensidad')
plt.legend()
plt.show()
print(A)
print(B)

plt.figure()
plt.scatter(x1,y1,label='Experimental 1',color='green',marker='x')
plt.plot(xn,yn,label='Ajuste 1')
#plt.plot(x,y,label='Ajuste. A='+str(round(A,2))+' B='+str(round(B,2)))
plt.xlabel('Ángulo en radianes')
plt.ylabel('Intensidad')
plt.scatter(x2,y2,label='Experimental 2',color='blue',marker='*')
plt.plot(x,y,label='Ajuste 2')
plt.title("Datos ajustados")
plt.legend()
plt.show()