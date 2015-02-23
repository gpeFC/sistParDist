"""
Modulo que contiene las funciones necesarias para calcular los valores
necesarios para ponderar y ajustar los parametros variables de la red.
"""

import math
import random

def pseudoaleatorio(inf, sup):
	"""
	"""
	while True:
		num_psa = random.uniform(inf, sup)
		if num_psa != 0.0:
			return num_psa

def suma_ponderada(bias, entrada, pesos):
	"""
	bias        Bias(umbral) sinaptico de la neurona.
	
	entrada     Entrada presinaptica de la neurona.
	
	pesos       Pesos sinapticos de la neurona.

	(Regla de propagacion neuronal) Regresa el valor del potencial sinaptico
	de cada neurona de la red.
	"""
	potencial = 0.0
	for i in range(len(pesos)):
		potencial += (pesos[i] * entrada[i])
	potencial += bias
	return potencial

def identidad_lineal(potencial):
	"""
	potencial        Potencial sinaptico de la neurona.

	Regresa el valor de activacion postsinaptico de la neurona mediante la
	funcion de transferencia identidad lineal.
	"""
	return potencial

def sigmoide_logistico(potencial):
	"""
	potencial        Potencial sinaptico de la neurona.

	Regresa el valor de activacion postsinaptico de la neurona mediante la
	funcion de transferencia sigmoide logistico.
	"""
	return (1.0 / (1.0 + (pow(e, -1.0 * potencial))))

def sigmoide_tangencial(potencial):
	"""
	potencial        Potencial sinaptico de la neurona.

	Regresa el valor de activacion postsinaptico de la neurona mediante la
	funcion de transferencia sigmoide tangencial.
	"""
	return ((2.0 / (1.0 + pow(e, -1.0 * pow))) - 1.0)

def sigmoide_hiperbolico(potencial):
	"""
	potencial        Potencial sinaptico de la neurona.

	Regresa el valor de activacion postsinaptico de la neurona mediante la
	funcion de transferencia sigmoide hiperbolico.
	"""
	return ((pow(e, potencial) - (pow(e, -1.0 * potencial))) / (pow(e, potencial) + (pow(e, -1.0 * potencial))))

def derivada_lineal(potencial):
	"""
	potencial        Potencial sinaptico de la neurona.

	Regresa el valor de la derivada del valor de activacion postsinaptico de
	la neurona para la funcion de transferencia identidad lineal.
	"""
	return 1.0

def derivada_logistica(potencial):
	"""
	potencial        Potencial sinaptico de la neurona.

	Regresa el valor de la derivada del valor de activacion postsinaptico de
	la neurona para la funcion de transferencia sigmoide logistico.
	"""
	return (sigmoide_logistico(potencial) * (1.0 - sigmoide_logistico(potencial)))

def derivada_tangencial(potencial):
	"""
	potencial        Potencial sinaptico de la neurona.

	Regresa el valor de la derivada del valor de activacion postsinaptico de
	la neurona para la funcion de transferencia sigmoide tangencial.
	"""
	return ((2.0 * pow(e, -1 * potencial)) / pow(1.0 + pow(e, -1 * pow), 2.0))

def derivada_hiperbolica(potencial):
	"""
	potencial        Potencial sinaptico de la neurona.

	Regresa el valor de la derivada del valor de activacion postsinaptico de
	la neurona para la funcion de transferencia sigmoide hiperbolico.
	"""
	return (1.0 - pow(sigmoide_hiperbolico(potencial), 2.0))
