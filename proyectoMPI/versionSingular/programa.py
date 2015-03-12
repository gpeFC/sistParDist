from random import shuffle
from pmcr import *

def algoritmo_retropropagacion(epocas, error, patrones, red):
	entradas = []
	salidas = []
	for patron in patrones:
		parametros = []
		for item in patron:
			if type(item) == list:
				salidas.append(item)
			else:
				parametros.append(item)
		entradas.append(parametros)
	iteracion = 0
	error_global = 0.0
	indices = range(len(patrones))
	while iteracion < epocas:
		print "Epoca:", iteracion+1
		error_global = 0.0
		shuffle(indices)
		for i in indices:
			red.propagacion(entradas[i])
			error_local = 0.0
			for j in range(len(salidas[i])):
				error_local += ((salidas[i][j] - red.capas[-1].neuronas[j].salida)**2)
			error_local /= float(len(red.capas[-1].neuronas))
			error_global += error_local
			if error_local != 0.0:
				red.retropropagacion(salidas[i], entradas[i])
			red.ajustar_parametros(entradas[i])
		error_global /= float(len(entradas))
		iteracion += 1
		if error_global <= error:
			break
	print "+"*30
	print "Epocas de entrenamiento:     ", iteracion
	print "Error minimo acordado:       ", error 
	print "Error minimo alcanzado:      ", error_global
	print "+"*30

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

error = 0.0005

epocas = 5

indices = [[2,2],[1]]


print
print "Ejemplo: Red Neuronal Perceptron Multicapa"
print 

red = RedNeuronal(1, 1, indices)

imprime_red(red)
print

algoritmo_retropropagacion(epocas, error, patrones, red)

imprime_red(red)
print