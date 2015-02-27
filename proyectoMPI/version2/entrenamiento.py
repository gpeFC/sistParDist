"""
Modulo que contiene los algoritmos de entrenamiento/aprendizaje para redes
perceptron multicapa.
"""

from random import shuffle
from perceptron import *


def algoritmo_retropropagacion(epocas, error, patrones, red_neural):
	"""
	"""
	iteracion = 0
	error_final = 0.0
	indices = range(len(patrones))
	while iteracion < epocas:
		error_global = 0.0
		shuffle(indices)
		for i in indices:
			error_local = 0.0
			entrada_patron = []
			for j in range(len(patrones[i])):
				if type(patrones[i][j]) != list:
					entrada_patron.append(patrones[i][j])
			red_neural.realizar_propagacion(entrada_patron)
			neuronas_salida = red_neural.capas[-1].neuronas
			for j in range(len(patrones[i][-1])):
				error_local += (patrones[i][-1][j] - neuronas_salida[j].salida)**2
			error_local /= float(len(neuronas_salida))
			error_global += error_local
			if error_local != 0.0:
				red_neural.realizar_retropropagacion(patrones[i][-1], entrada_patron)
			red_neural.actualizar_parametros_neuronales(entrada_patron)
		error_global /= float(len(patrones))
		error_final = error_global
		iteracion += 1
		if error_global <= error:
			break
		print "\nError global:", error_global

	print "+"*30
	print "Epocas de entrenamiento:     ", iteracion
	print "Error minimo acordado:       ", error 
	print "Error minimo alcanzado:      ", error_final
	print "+"*30


def backpropagation(epocas, error, patrones, red):
	entradas = []
	salidas = []
	for patron in patrones:
		paramts = []
		for item in patron:
			if type(item) == list:
				salidas.append(item)
			else:
				paramts.append(item)
		entradas.append(paramts)
	iteracion = 0
	error_final = 0.0
	indices = range(len(patrones))
	while iteracion < epocas:
		error_global = 0.0
		shuffle(indices)
		for i in indices:
			error_local = 0.0
			red.realizar_propagacion(entradas[i])
			for j in range(len(salidas[i])):
				error_local += ((salidas[i][j] - red.capas[-1].neuronas[j].salida)**2)
			error_local /= float(len(red.capas[-1].neuronas))
			print "Error local:", error_local
			error_global += error_local
			if error_local != 0.0:
				red.realizar_retropropagacion(salidas[i], entradas[i])
			red.actualizar_parametros_neuronales(entradas[i])
		error_global /= float(len(entradas))
		error_final = error_global
		iteracion += 1
		if error_global <= error:
			break
		print "Epoca:", iteracion
		for i in range(len(red.capas)):
			for j in range(len(red.capas[i].neuronas)):
				print "\tBias:", red.capas[i].neuronas[j].bias
				print "\tAlpha:", red.capas[i].neuronas[j].alpha
				print "\tPesos:", red.capas[i].neuronas[j].pesos
			print

	print "+"*30
	print "Epocas de entrenamiento:     ", iteracion
	print "Error minimo acordado:       ", error 
	print "Error minimo alcanzado:      ", error_final
	print "+"*30