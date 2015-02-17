"""
Modulo que contiene las clases necesarias para crear objetos de redes
neuronales artificiales de tipo perceptron multicapa y ser entrenadas
con un algoritmo de retropropagacion.
"""

import random

class Neurona:
	"""
	Neurona artificial tipo perceptron.
	"""

	def __init__(self, total_args):
		"""
		total_args        Numero total de valores de los pesos sinapticos
		                  de la neurona.

		Inicializa los datos miembro de la neurona.
		"""
		self.__alpha = 0.0
		self.__salida = 0.0
		while True:
			self.__bias = random.random()
			if self.__bias != 0.0:
				break
			else:
				continue
		for i in range(total_args):
			while True:
				peso = random.random()
				if peso != 0.0:
					self.__pesos.append(peso)
					break
				else:
					continue

	def establecer_alpha(self, alpha):
		"""
		alpha        Factor de aprendizaje de la neurona.

		Establece el valor del factor de aprendizaje de la neurona.
		"""
		self.__alpha = alpha

	def establecer_bias(self, bias):
		"""
		bias        Bias(umbral) sinaptico de la neurona.

		Establece el valor del bias sinaptico de la neurona.
		"""
		self.__bias = bias

	def establecer_pesos(self, pesos):
		"""
		pesos        Pesos sinapticos de la neurona.

		Establece los valores de los pesos sinapticos de la neurona.
		"""
		self.__pesos = pesos

	def obtener_alpha(self):
		"""
		Regresa el valor del alpha de la neurona.
		"""
		return self.__alpha

	def obtener_bias(self):
		"""
		Regresa el valor del bias de la neurona.
		"""
		return self.__bias

	def obtener_pesos(self):
		"""
		Regresa los valores de los pesos sinapticos de la neurona.
		"""
		return self.__pesos

	def obtener_salida(self):
		"""
		Regresa el valor de la salida postsinaptica de la neurona.
		"""
		return self.__salida

	def calcular_salida(self, id_funcion, entrada):
		"""
		id_funcion        Indice de la funcion de activacion asociada a la 
		                  neurona.

		entrada           Entrada presinaptica de la neurona.

		Calcula la salida postsinaptica de la neurona.
		"""
		if id_funcion == 1:
			self.__salida = identidad_lineal(suma_ponderada(self.__bias, entrada, self.__pesos))
		elif id_funcion == 2:
			self.__salida = sigmoide_logistico(suma_ponderada(self.__bias, entrada, self.__pesos))
		elif id_funcion == 3:
			self.__salida = sigmoide_tangencial(suma_ponderada(self.__bias, entrada, self.__pesos))
		elif id_funcion == 4:
			self.__salida = sigmoide_hiperbolico(suma_ponderada(self.__bias, entrada, self.__pesos))

class CapaNeuronal:
	"""
	Capa de neuronas artificiales tipo perceptron.
	"""

	def __init__(self, total_neurs, total_args):
		"""
		total_neurs        Numero total de neuronas de la capa.

		total_args        Numero total de valores de los pesos sinapticos
		                  de cada neurona de la capa.

		Inicializa los datos miembro de la capa.
		"""
		#self.__funciones = []
		self.__delthas = [0.0] * total_neurs
		#self.__salidas = []
		#self.__entradas = []
		self.__neuronas = []
		for i in range(total_neurs):
			self.__neuronas.append(Neurona(total_args))

	#def establecer_funciones(self):
	#	pass

	#def establecer_entradas(self):
	#	pass

	def establecer_alphas(self):
		pass

	#def obtener_funciones(self):
	#	pass

	def obtener_delthas(self):
		pass

	def obtener_salidas(self):
		pass

	def obtener_pesos(self):
		pass

	def obtener_neuronas(self):
		pass

	def actualizar_biases(self):
		pass

	def actualizar_pesos(self):
		pass

	def calcular_delthas(self):
		pass

	def calcular_salidas(self):
		pass

class RedNeuronal:
	"""
	Red neuronal artificial tipo perceptron multicapa.
	"""

	def __init__(self):
		"""
		"""
		self.__nombre_red = ""
		self.__configuracion_alphas = ""
		self.__configuracion_funciones = ""
		self.__total_neuronas_capa = []
		self.__indice_funcion_activacion = []
		self.__red_neuronal = []

	def establecer_nombre_red(self):
		pass

	def establecer_configuracion_alphas(self):
		pass

	def establecer_configuracion_funciones(self):
		pass

	def establecer_indice_funcion_activacion(self):
		pass

	def establecer_red_neuronal(self):
		pass

	def obtener_nombre_red(self):
		pass

	def obtener_red_neuronal(self):
		pass

	def aplicar_red_neuronal(self):
		pass

	def realizar_propagacion(self):
		pass

	def realizar_retropropagacion(self):
		pass

	def actualizar_parametros_neuronales(self):
		pass