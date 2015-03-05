"""
Modulo que contiene los algoritmos de entrenamiento/aprendizaje para redes
perceptron multicapa.
"""

import math
from random import shuffle
from perceptron import *

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
		print "Epoca:", iteracion+1
		error_global = 0.0
		shuffle(indices)
		for i in indices:
			print "\tPatron:", entradas[i]
			print "\tSalida-Obj:", salidas[i]
			red.realizar_propagacion(entradas[i])
			error_local = 0.0
			for j in range(len(salidas[i])):
				print "\t   Salida-Obt:", red.capas[-1].neuronas[j].salida
				print "\t   Error:", (salidas[i][j] - red.capas[-1].neuronas[j].salida)
				error_local += pow((salidas[i][j] - red.capas[-1].neuronas[j].salida),2)
			error_local /= float(len(red.capas[-1].neuronas))
			#print "Error local:", error_local
			error_global += error_local
			if error_local != 0.0:
				red.realizar_retropropagacion(salidas[i], entradas[i])
			red.actualizar_parametros_neuronales(entradas[i])
		error_global /= float(len(entradas))
		error_final = error_global
		iteracion += 1
		if error_global <= error:
			break
		
		entrar = raw_input("<Enter>")
		imprime_red(red)
		entrar = raw_input("<Enter>")

	print "+"*30
	print "Epocas de entrenamiento:     ", iteracion
	print "Error minimo acordado:       ", error 
	print "Error minimo alcanzado:      ", error_final
	print "+"*30