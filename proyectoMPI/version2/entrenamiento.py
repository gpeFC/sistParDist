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
		print "Entrenamiento en epoca: ", iteracion
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

	print "+"*30
	print "Epocas de entrenamiento:     ", interacion
	print "Error minimo acordado:       ", error 
	print "Error minimo alcanzado:      ", error_final
	print "+"*30
