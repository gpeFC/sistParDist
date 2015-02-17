import math

def suma_ponderada(bias, entrada, pesos):
	"""
	"""
	potencial = 0.0
	for i in range(len(pesos)):
		potencial += (pesos[i] * entrada[i])
	potencial += bias
	return potencial

def identidad_lineal(potencial):
	"""
	"""
	return potencial

def sigmoide_logistico(potencial):
	"""
	"""
	return (1.0 / (1.0 + (pow(e, -1 * potencial))))

def sigmoide_tangencial(potencial):
	"""
	"""
	return ((2.0 / (1.0 + pow(e, -1 * pow))) - 1.0)

def sigmoide_hiperbolico(potencial):
	"""
	"""
	return ((pow(e, potencial) - (pow(e, -1 * potencial))) / (pow(e, potencial) + (pow(e, -1 * potencial))))

def derivada_lineal(potencial):
	"""
	"""
	return 1.0

def derivada_logistica(potencial):
	"""
	"""
	return (sigmoide_logistico(potencial) * (1.0 - sigmoide_logistico(potencial)))

def derivada_tangencial(potencial):
	"""
	"""
	return ((2.0 * pow(e, -1 * potencial)) / pow(1.0 + pow(e, -1 * pow), 2))

def derivada_hiperbolica(potencial):
	"""
	"""
	return (1.0 - pow(sigmoide_hiperbolico(potencial), 2))