import math
import random

def pseudoaleatorio(inf, sup):
	while True:
		num_psa = random.uniform(inf, sup)
		if num_psa != 0.0:
			return num_psa

def suma_ponderada(bias, entrada, pesos):
	potencial = 0.0
	for i in range(len(pesos)):
		potencial += (pesos[i] * entrada[i])
	potencial += bias
	return potencial

def identidad_lineal(potencial):
	return potencial

def sigmoide_logistico(potencial):
	return (1.0 / (1.0 + (pow(math.e, (-1.0 * potencial)))))

def sigmoide_tangencial(potencial):
	return ((2.0 / (1.0 + pow(math.e, (-1.0 * potencial)))) - 1.0)

def sigmoide_hiperbolico(potencial):
	return ((pow(math.e, potencial) - (pow(math.e, (-1.0 * potencial)))) / (pow(math.e, potencial) + (pow(math.e, (-1.0 * potencial)))))

def derivada_lineal(potencial):
	return 1.0

def derivada_logistica(potencial):
	return (sigmoide_logistico(potencial) * (1.0 - sigmoide_logistico(potencial)))

def derivada_tangencial(potencial):
	return ((2.0 * pow(math.e, (-1 * potencial))) / pow(1.0 + pow(math.e, (-1 * potencial)), 2.0))

def derivada_hiperbolica(potencial):
	return (1.0 - pow(sigmoide_hiperbolico(potencial), 2.0))
