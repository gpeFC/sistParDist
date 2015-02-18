"""
Modulo que contiene las clases necesarias para crear objetos de redes
neuronales artificiales de tipo perceptron multicapa y ser entrenadas
con un algoritmo de retropropagacion.
"""

import random
import funciones

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
		"""
		Regresa los valores de los errores cometidos por cada neurona de la
		capa en la propagacion actual.
		"""

	#def obtener_salidas(self):
	#	pass

	#def obtener_pesos(self):
	#	pass

	def obtener_neuronas(self):
		"""
		Regresa el vector de neuronas de la capa.
		"""

	def actualizar_biases(self):
		"""
		Actualiza el bias de cada neurona de la capa.
		"""
		for i in range(len(self.__neuronas)):
			bias_actual = self.__neuronas[i].obtener_bias()
			bias_nuevo = bias_actual + (self.__neuronas[i].obtener_alpha() * self.__delthas[i])
			self.__neuronas[i].establecer_bias(bias_nuevo)

	def actualizar_pesos(self, entrada):
		"""
		entrada        Entrada presinaptica de cada neurona de la capa.

		Actualiza los pesos sinapticos de cada neurona de la capa.
		"""
		for i in range(len(self.__neuronas)):
			pesos_actuales = self.__neuronas[i].obtener_pesos()
			pesos_nuevos = [0.0] * len(entrada)
			for j in range(len(entrada)):
				pesos_nuevos[j] = pesos_actuales[j] + (self.__neuronas[i].obtener_alpha() * self.__delthas[i] * entrada[j])
			self.__neuronas[i].establecer_pesos(pesos_nuevos)

	def calcular_delthas_salida(self, id_funciones, errores, entrada):
		"""
		id_funciones        Vector de indices que indican la funcion de 
		                    activacion asociada a la neurona.

		errores             Vector de errores obtenidos en las neuronas
		                    de la capa de salida de la red.

		entrada             Vector de valores presinapticos de entrada de
		                    la capa.

		Calcula los errores deltha cometidos por cada neurona de la capa
		de salida de la red.
		"""
		for i in range(len(id_funciones)):
			if id_funciones[i] == 1:
				self.__delthas[i] = errores[i] * derivada_lineal(suma_ponderada(self.__neuronas[i].obtener_bias(), entrada, self.__neuronas[i].obtener_pesos()))
			elif id_funciones[i] == 2:
				self.__delthas[i] = errores[i] * derivada_logistica(suma_ponderada(self.__neuronas[i].obtener_bias(), entrada, self.__neuronas[i].obtener_pesos()))
			elif id_funciones[i] == 3:
				self.__delthas[i] = errores[i] * derivada_tangencial(suma_ponderada(self.__neuronas[i].obtener_bias(), entrada, self.__neuronas[i].obtener_pesos()))
			elif id_funciones[i] == 4:
				self.__delthas[i] = errores[i] * derivada_hiperbolica(suma_ponderada(self.__neuronas[i].obtener_bias(), entrada, self.__neuronas[i].obtener_pesos()))

	def calcular_delthas_ocultas(self, id_funciones, delthas, entrada, neuronas):
		"""
		id_funciones        Vector de indices que indican la funcion de 
		                    activacion asociada a la neurona.

		delthas        Vector de errores deltha calculados en la capa 
		               posterior.

		entrada             Vector de valores presinapticos de entrada de
		                    la capa.

		neuronas       Vector de neuronas de la capa posterior.

		Calcula los errores deltha cometidos por cada neurona de las capas
		ocultas de la red.
		"""
		for i in range(len(self.__neuronas)):
			suma_deltha = 0.0
			for j in range(len(neuronas)):
				pesos = neuronas[j].obtener_pesos()
				suma_deltha += (delthas[j] * pesos[i])
			if id_funciones[i] == 1:
				self.__delthas[i] = derivada_lineal(suma_ponderada(self.__neuronas[i].obtener_bias(), entrada, self.__neuronas[i].obtener_pesos())) * suma_deltha
			elif id_funciones[i] == 2:
				self.__delthas[i] = derivada_logistica(suma_ponderada(self.__neuronas[i].obtener_bias(), entrada, self.__neuronas[i].obtener_pesos())) * suma_deltha
			elif id_funciones[i] == 3:
				self.__delthas[i] = derivada_tangencial(suma_ponderada(self.__neuronas[i].obtener_bias(), entrada, self.__neuronas[i].obtener_pesos())) * suma_deltha
			elif id_funciones[i] == 4:
				self.__delthas[i] = derivada_hiperbolica(suma_ponderada(self.__neuronas[i].obtener_bias(), entrada, self.__neuronas[i].obtener_pesos())) * suma_deltha

	def calcular_salidas(self, id_funciones, entrada):
		"""
		id_funciones        Vector de indices que indican la funcion de 
		                    activacion asociada a la neurona.

		entrada             Vector de valores presinapticos de entrada de
		                    la capa.

		Calcula las salidas postsinapticas de cada neurona de la capa.
		"""
		for i in range(len(id_funciones)):
			self.__neuronas[i].calcular_salida(id_funciones[i], entrada)

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

	def establecer_nombre_red(self, nombre_red):
		"""
		"""
		self.__nombre_red = nombre_red

	def establecer_configuracion_alphas(self, config_alphas):
		"""
		"""
		self.__configuracion_alphas = config_alphas

	def establecer_configuracion_funciones(self, config_funciones):
		"""
		"""
		self.__configuracion_funciones = config_funciones

	def establecer_indice_funcion_activacion(self, indice_func_actv):
		"""
		"""
		self.__indice_funcion_activacion = indice_func_actv

	def establecer_red_neuronal(self, total_args, total_capas, total_neurs_capas):
		"""
		"""
		for i in range(total_capas):
			if i == 0:
				self.__red_neuronal.append(CapaNeuronal(total_neurs_capas[i], total_args))
			else:
				self.__red_neuronal.append(CapaNeuronal(total_neurs_capas[i], total_neurs_capas[i-1]))

	def obtener_nombre_red(self):
		"""
		"""
		return self.__nombre_red

	def obtener_red_neuronal(self):
		"""
		"""
		return self.__red_neuronal

	def aplicar_red_neuronal(self):
		pass

	def realizar_propagacion(self):
		pass

	def realizar_retropropagacion(self):
		pass

	def actualizar_parametros_neuronales(self):
		pass