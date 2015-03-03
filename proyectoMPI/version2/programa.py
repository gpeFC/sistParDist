"""
"""

import math
from perceptron import *
from entrenamiento import *


def imprime_red(red):
	print "="*60
	for i in range(len(red.capas)):
		print "Capa " + str(i+1) + "-"*35
		for j in range(len(red.capas[i].neuronas)):
			print "Neurona " + str(j+1)
			print "\tBias:", red.capas[i].neuronas[j].bias
			print "\tAlpha:", red.capas[i].neuronas[j].alpha
			print "\tPesos:", red.capas[i].neuronas[j].pesos
	print "="*60



patrones = [
			[-2.0,[-1.0]],
			[-1.2,[-0.81]],
			[0.4,[-0.31]],
			[0.4,[0.309]],
			[1.2,[0.809]],
			[2.0,[1.0]],
			]
ejemplos = [
			[-1.0],
			[-0.5],
			[0.4],
			[1.0],
			[1.6],
			]

error = 0.0
for i in range(len(patrones)):
	error += patrones[i][1][0]
error /= float(len(patrones))
error *= (35.0/100.0)
error = abs(error)

error = 0.0005

epocas = 30

indices = [[3,3],[1]]

print
print "Ejemplo: Red Neuronal Perceptron Multicapa"
print 

red = RedNeuronal(1,"RED1","TDA/RED","FNCN/SALIDAOCULTAS",indices)

red.establecer_valores_alphas(1)

imprime_red(red)
print 

backpropagation(epocas,error,patrones,red)

print 
imprime_red(red)
print 

"""
print
for i in range(len(red.capas)):
	print red.capas[i].delthas
print
"""