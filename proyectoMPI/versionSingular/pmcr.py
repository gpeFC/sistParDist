"""
Modulo para crear redes neuronales artificiales tipo perceptron multicapa y
ser entrenadas con el algoritmo de retropropagacion.
"""


import math
import random
from random import shuffle


def pseudoaleatorio(inferior, superior):
	"""
	Regresa un numero real pseudoaleatorio de un rango definido, el numero
	es distinto de las cotas del rango.

	inferior   Cota inferior del rango.
	superior   Cota superior del rango.
	"""
	while True:
		numero = random.uniform(inferior, superior)
		if numero != inferior and numero != superior:
			return numero


def suma_ponderada(bias, entrada, pesos):
	"""
	(Regla de propagacion) Regresa el potencial sinaptico de una neurona.

	bias      Bias/umbral sinaptico de la neurona.
	entrada   Vector de valores presinapticos de la neurona.
	pesos     Vector de pesos sinapticos de la neurona.
	"""
	potencial = 0.0
	for i in range(len(pesos)):
		potencial += (pesos[i] * entrada[i])
	potencial += bias
	return potencial


def activacion(id_funcion, potencial):
	"""
	Regresa la activacion postsinaptica de la neurona.

	id_funcion   Indice que indica la funcion de activacion de la neurona.
	potencial    Potencial sinaptico de la neurona.
	"""
	if id_funcion == 1:
		return potencial
	elif id_funcion == 2:
		return (1.0 / (1.0 + (pow(math.e, (-1.0 * potencial)))))
	elif id_funcion == 3:
		return ((2.0 / (1.0 + pow(math.e, (-1.0 * potencial)))) - 1.0)
	elif id_funcion == 4:
		return ((pow(math.e, potencial) - (pow(math.e, (-1.0 * potencial)))) / (pow(math.e, potencial) + (pow(math.e, (-1.0 * potencial)))))


def derivada(id_funcion, potencial):
	"""
	Regresa la derivada de la activacion postsinaptica de la neurona.

	id_funcion   Indice que indica la funcion de activacion de la neurona.
	potencial    Potencial sinaptico de la neurona.
	"""
	if id_funcion == 1:
		return 1.0
	elif id_funcion == 2:
		return (activacion(2, potencial) * (1.0 - activacion(2, potencial)))
	elif id_funcion == 3:
		return ((2.0 * pow(math.e, (-1 * potencial))) / pow(1.0 + pow(math.e, (-1 * potencial)), 2.0))
	elif id_funcion == 4:
		return (1.0 - pow(activacion(4, potencial), 2.0))


def algoritmo_retropropagacion(epocas, error, patrones, red):
	"""
	Algoritmo de retropropagacion estandar para entrenar redes multicapa 
	supervisadas mediante la minimizacion del error cuadratico medio cometido
	para cada patron de entrenamiento.

	epocas     Total de epocas/iteraciones de entrenamiento de la red.
	error      Error medio general de los patrones de entrenamiento acordado.
	patrones   Patrones de entrenamiento de la red.
	red        Red neuronal a ser entrenada.
	"""
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
		error_global = 0.0
		shuffle(indices)
		for i in indices:
			red.propagacion(entradas[i])
			error_local = 0.0
			for j in range(len(salidas[i])):
				error_local += ((salidas[i][j] - red.capas[-1].salidas[j])**2)
			error_local /= float(len(red.capas[-1].salidas))
			error_global += error_local
			if error_local != 0.0:
				red.retropropagacion(salidas[i])
				red.ajustar_parametros()
		error_global /= float(len(entradas))
		iteracion += 1
		if error_global <= error:
			break
	print "+"*30
	print "Epocas de entrenamiento:     ", iteracion
	print "Error minimo acordado:       ", error 
	print "Error minimo alcanzado:      ", error_global
	print "+"*30


class Neurona(object):
	"""Neurona artificial tipo Perceptron."""

	def __init__(self, total_args):
		"""
		Inicializa la neurona.

		total_args   Total de valores en el vector de pesos sinapticos.
		"""
		super(Neurona, self).__init__()
		self.alpha = 0.0
		self.salida = 0.0
		self.bias = pseudoaleatorio(-1.0, 1.0)
		self.pesos = []
		for i in range(total_args):
			self.pesos.append(pseudoaleatorio(-1.0, 1.0))

	def calcular_salida(self, id_funcion, entrada):
		"""
		Calcula la salida postsinaptica de la neurona.

		id_funcion   Indice que indica la funcion de activacion de la neurona.
		entrada      Vector de valores presinapticos de la neurona.
		"""
		self.salida = activacion(id_funcion, suma_ponderada(self.bias, entrada, self.pesos))


class CapaNeuronal(object):
	"""Capa de neuronas artificiales tipo Perceptron."""

	def __init__(self, total_neurs, total_args, indice_funciones):
		"""
		Inicializa la capa.

		total_neurs        Total de neuronas en la capa.
		total_args         Total de valores en el vector de pesos sinapticos
		                   de cada neurona de la capa.
		indice_funciones   Indices que indican la funcion de activacion
		                   asociada a cada neurona de la capa.
		"""
		super(CapaNeuronal, self).__init__()
		self.funciones = indice_funciones
		self.delthas = [0.0] * total_neurs
		self.salidas = [0.0] * total_neurs
		self.entrada = []
		self.neuronas = []
		for i in range(total_neurs):
			self.neuronas.append(Neurona(total_args))

	def ajustar_biases(self):
		"""
		Ajusta/actualiza el valor del bias/umbral de cada neurona de la capa
		durante el entrenamiento de la red.
		"""
		for i in range(len(self.neuronas)):
			self.neuronas[i].bias += (self.neuronas[i].alpha * self.delthas[i])

	def ajustar_pesos(self):
		"""
		Ajusta/actualiza el valor de los pesos sinapticos de cada neurona de 
		la capa durante el entrenamiento de la red.
		"""
		for i in range(len(self.neuronas)):
			for j in range(len(self.neuronas[i].pesos)):
				self.neuronas[i].pesos[j] += (self.neuronas[i].alpha * self.delthas[i] * self.entrada[j])

	def calcular_delthas_salida(self, errores):
		"""
		Calcula los errores delta cometidos por cada una de las neuronas de 
		la capa de salida durante el entrenamiento de la red.

		errores   Vector de errores cometidos por cada neurona de la capa.
		"""
		for i in range(len(errores)):
			self.delthas[i] = errores[i] * derivada(self.funciones[i],
													suma_ponderada(self.neuronas[i].bias,
													self.entrada, self.neuronas[i].pesos))

	def calcular_delthas_ocultas(self, capa_sig):
		"""
		Calcula los errores delta cometidos por cada una de las neuronas de 
		las capas ocultas durante el entrenamiento de la red.

		capa_sig   Capa de neuronas siguiente a la capa actual.
		"""
		for i in range(len(self.funciones)):
			suma_delthas = 0.0
			for j in range(len(capa_sig.delthas)):
				suma_delthas += (capa_sig.delthas[j] * capa_sig.neuronas[j].pesos[i])
			self.delthas[i] = suma_delthas * derivada(self.funciones[i], suma_ponderada(
													self.neuronas[i].bias, self.entrada,
													self.neuronas[i].pesos))

	def calcular_salidas(self):
		"""
		Calcula las salidas postsinapticas de las neuronas de la capa.
		"""
		for i in range(len(self.funciones)):
			self.neuronas[i].calcular_salida(self.funciones[i], self.entrada)
			self.salidas[i] = self.neuronas[i].salida


class RedNeuronal(object):
	"""Red de neuronas artificiales tipo Perceptron."""

	def __init__(self, id_alphas, total_args, indice_funciones):
		"""
		Inicializa la red.

		id_alphas          Indice que indica el tipo de configuracion del 
		                   factor de aprendizaje (red/capa/neurona).
		total_args         Total de valores en el vector de pesos sinapticos
		                   de cada neurona de la primer capa oculta.
		indice_funciones   Indices que indican la funcion de activacion
		                   asociada a cada neurona de la red.
		"""
		super(RedNeuronal, self).__init__()
		self.capas = []
		for i in range(len(indice_funciones)):
			if i == 0:
				self.capas.append(CapaNeuronal(len(indice_funciones[i]), total_args, indice_funciones[i]))
			else:
				self.capas.append(CapaNeuronal(len(indice_funciones[i]), len(indice_funciones[i-1]), indice_funciones[i]))
		if id_alphas == 1:
			alpha = pseudoaleatorio(0.0, 1.0)
			for i in range(len(self.capas)):
				for j in range(len(self.capas[i].neuronas)):
					self.capas[i].neuronas[j].alpha = alpha
		elif id_alphas == 2:
			for i in range(len(self.capas)):
				alpha = pseudoaleatorio(0.0, 1.0)
				for j in range(len(self.capas[i].neuronas)):
					self.capas[i].neuronas[j].alpha = alpha
		elif id_alphas == 3:
			for i in range(len(self.capas)):
				for j in range(len(self.capas[i].neuronas)):
					self.capas[i].neuronas[j].alpha = pseudoaleatorio(0.0, 1.0)

	def aplicar_red(self, entradas):
		"""
		Regresa las salidas calculadas por la red neuronal entrenada para los
		valores que le sean ingresados.

		entradas   Vector de valores ingresados a la red para ser procesados.
		"""
		salidas = []
		for i in range(len(entradas)):
			self.propagacion(entradas[i])
			salida = []
			for j in range(len(self.capas[-1].neuronas)):
				salida.append(self.capas[-1].neuronas[j].salida)
			salidas.append(salida)
		return salidas

	def propagacion(self, entrada):
		"""
		Propaga a traves de todas las capas de la red un vector de valores
		ingresado a la red.

		entrada   Vector de valores ingresado a la red para ser procesado.
		"""
		for i in range(len(self.capas)):
			if i == 0:
				self.capas[i].entrada = entrada
			else:
				entrada = self.capas[i-1].salidas
				self.capas[i].entrada = entrada
			self.capas[i].calcular_salidas()

	def retropropagacion(self, salida):
		"""
		Propaga hacia atras, de la ultima capa a la primera, en la red los 
		errores cometidos al procesar los valores ingresados a la red durante
		su entrenamiento.

		salida   Vector de salidas reales de los valores ingresados a la red.
		"""
		lista = range(len(self.capas))
		for i in lista.__reversed__():
			if i == len(self.capas) - 1:
				errores = []
				for j in range(len(self.capas[i].salidas)):
					errores.append((salida[j] - self.capas[i].salidas[j]))
				self.capas[i].calcular_delthas_salida(errores)
			else:
				capa_siguiente = self.capas[i+1]
				self.capas[i].calcular_delthas_ocultas(capa_siguiente)

	def ajustar_parametros(self):
		"""
		Ajusta/actualiza los valores de los parametros de la red durante el
		entrenamiento.
		"""
		for i in range(len(self.capas)):
			self.capas[i].ajustar_biases()
			self.capas[i].ajustar_pesos()

	def imprime_red(self):
		"""
		Imprime en pantalla los valores de los parametros de la red.
		"""
		print "="*60
		for i in range(len(self.capas)):
			print "Capa " + str(i+1) + "-"*35
			for j in range(len(self.capas[i].neuronas)):
				print "Neurona " + str(j+1)
				print "\tBias:", self.capas[i].neuronas[j].bias
				print "\tAlpha:", self.capas[i].neuronas[j].alpha
				print "\tPesos:", self.capas[i].neuronas[j].pesos
		print "="*60
		