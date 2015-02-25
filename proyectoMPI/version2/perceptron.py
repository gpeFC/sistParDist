"""
Modulo que contiene las clases necesarias para crear objetos de redes
neuronales artificiales de tipo perceptron multicapa y ser entrenadas
con un algoritmo de retropropagacion.
"""


from funciones import *


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
		self.alpha = 0.0
		self.salida = 0.0
		self.bias = pseudoaleatorio(-1.0, 1.0)
		self.pesos = []
		for i in range(total_args):
			self.pesos.append(pseudoaleatorio(-1.0, 1.0))

	def calcular_salida(self, id_funcion, entrada):
		"""
		id_funcion        Indice de la funcion de activacion asociada a la 
		                  neurona.

		entrada           Entrada presinaptica de la neurona.

		Calcula la salida postsinaptica de la neurona.
		"""
		if id_funcion == 1:
			self.salida = identidad_lineal(suma_ponderada(self.bias, entrada, self.pesos))
		elif id_funcion == 2:
			self.salida = sigmoide_logistico(suma_ponderada(self.bias, entrada, self.pesos))
		elif id_funcion == 3:
			self.salida = sigmoide_tangencial(suma_ponderada(self.bias, entrada, self.pesos))
		elif id_funcion == 4:
			self.salida = sigmoide_hiperbolico(suma_ponderada(self.bias, entrada, self.pesos))


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
		self.delthas = [0.0] * total_neurs
		self.neuronas = []
		for i in range(total_neurs):
			self.neuronas.append(Neurona(total_args))

	def establecer_alphas(self, alphas):
		"""
		"""
		for i in range(len(alphas)):
			self.neuronas[i].establecer_alpha(alphas[i])

	def actualizar_biases(self):
		"""
		Actualiza el bias de cada neurona de la capa.
		"""
		for i in range(len(self.neuronas)):
			bias_actual = self.neuronas[i].obtener_bias()
			bias_nuevo = bias_actual + (self.neuronas[i].obtener_alpha() * self.delthas[i])
			self.neuronas[i].establecer_bias(bias_nuevo)

	def actualizar_pesos(self, entrada):
		"""
		entrada        Entrada presinaptica de cada neurona de la capa.

		Actualiza los pesos sinapticos de cada neurona de la capa.
		"""
		for i in range(len(self.neuronas)):
			pesos_actuales = self.neuronas[i].obtener_pesos()
			pesos_nuevos = [0.0] * len(entrada)
			for j in range(len(entrada)):
				pesos_nuevos[j] = pesos_actuales[j] + (self.neuronas[i].obtener_alpha() * self.delthas[i] * entrada[j])
			self.neuronas[i].establecer_pesos(pesos_nuevos)

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
				self.delthas[i] = errores[i] * derivada_lineal(suma_ponderada(self.neuronas[i].obtener_bias(), entrada, self.neuronas[i].obtener_pesos()))
			elif id_funciones[i] == 2:
				self.delthas[i] = errores[i] * derivada_logistica(suma_ponderada(self.neuronas[i].obtener_bias(), entrada, self.neuronas[i].obtener_pesos()))
			elif id_funciones[i] == 3:
				self.delthas[i] = errores[i] * derivada_tangencial(suma_ponderada(self.neuronas[i].obtener_bias(), entrada, self.neuronas[i].obtener_pesos()))
			elif id_funciones[i] == 4:
				self.delthas[i] = errores[i] * derivada_hiperbolica(suma_ponderada(self.neuronas[i].obtener_bias(), entrada, self.neuronas[i].obtener_pesos()))

	def calcular_delthas_ocultas(self, id_funciones, delthas, entrada, neuronas):
		"""
		id_funciones        Vector de indices que indican la funcion de 
		                    activacion asociada a la neurona.

		delthas        Vector de errores deltha calculados en la capa 
		               capa_siguiente.

		entrada             Vector de valores presinapticos de entrada de
		                    la capa.

		neuronas       Vector de neuronas de la capa capa_siguiente.

		Calcula los errores deltha cometidos por cada neurona de las capas
		ocultas de la red.
		"""
		for i in range(len(self.neuronas)):
			suma_deltha = 0.0
			for j in range(len(neuronas)):
				pesos = neuronas[j].obtener_pesos()
				suma_deltha += (delthas[j] * pesos[i])
			if id_funciones[i] == 1:
				self.delthas[i] = derivada_lineal(suma_ponderada(self.neuronas[i].obtener_bias(), entrada, self.neuronas[i].obtener_pesos())) * suma_deltha
			elif id_funciones[i] == 2:
				self.delthas[i] = derivada_logistica(suma_ponderada(self.neuronas[i].obtener_bias(), entrada, self.neuronas[i].obtener_pesos())) * suma_deltha
			elif id_funciones[i] == 3:
				self.delthas[i] = derivada_tangencial(suma_ponderada(self.neuronas[i].obtener_bias(), entrada, self.neuronas[i].obtener_pesos())) * suma_deltha
			elif id_funciones[i] == 4:
				self.delthas[i] = derivada_hiperbolica(suma_ponderada(self.neuronas[i].obtener_bias(), entrada, self.neuronas[i].obtener_pesos())) * suma_deltha

	def calcular_salidas(self, id_funciones, entrada):
		"""
		id_funciones        Vector de indices que indican la funcion de 
		                    activacion asociada a la neurona.

		entrada             Vector de valores presinapticos de entrada de
		                    la capa.

		Calcula las salidas postsinapticas de cada neurona de la capa.
		"""
		for i in range(len(id_funciones)):
			self.neuronas[i].calcular_salida(id_funciones[i], entrada)


class RedNeuronal:
	"""
	Red neuronal artificial tipo perceptron multicapa.
	"""

	def __init__(self, total_args, nombre, config_alphas, config_funcns, indices_funcns):
		"""
		"""
		self.nombre_red = nombre
		self.configuracion_alphas = config_alphas
		self.configuracion_funciones = config_funcns
		self.indice_funcion_activacion = indices_funcns
		self.red_neuronal = []
		for i in range(len(self.indice_funcion_activacion)):
			if i == 0:
				self.red_neuronal.append(CapaNeuronal(len(self.indice_funcion_activacion[i]), total_args))
			else:
				self.red_neuronal.append(CapaNeuronal(len(self.indice_funcion_activacion[i]),
					len(self.indice_funcion_activacion[i-1])))

	def establecer_valores_alphas(self, indice):
		"""
		"""
		if indice == 1:
			alpha = pseudoaleatorio(-1.0, 1.0)
			for i in range(len(self.red_neuronal)):
				alphas = [alpha] * len(self.indice_funcion_activacion[i])
				self.red_neuronal[i].establecer_alphas(alphas)
		elif indice == 2:
			for i in range(len(self.red_neuronal)):
				alpha = pseudoaleatorio(-1.0, 1.0)
				alphas = [alpha] * len(self.indice_funcion_activacion[i])
				self.red_neuronal[i].establecer_alphas(alphas)
		elif indice == 3:
			for i in range(len(self.red_neuronal)):
				alphas = []
				for j in range(len(self.indice_funcion_activacion[i])):
					alphas.append(pseudoaleatorio(-1.0, 1.0))
				self.red_neuronal[i].establecer_alphas(alphas)

	def aplicar_red_neuronal(self, entrada):
		"""
		"""
		salidas = []
		for i in range(len(entrada)):
			self.realizar_propagacion(entrada[i])
			neuronas_salida = self.red_neuronal[-1].neuronas
			salida = 0.0
			for j in range(len(neuronas_salida)):
				salida += neuronas_salida[j].salida
			salida /= float(len(neuronas_salida))
			salidas.append(salida)
		return salidas

	def realizar_propagacion(self, entrada):
		"""
		"""
		for i in range(len(self.red_neuronal)):
			capa = self.red_neuronal[i]
			if i == 0:
				capa.calcular_salidas(self.indice_funcion_activacion[i], entrada)
			else:
				neuronas = self.red_neuronal[i-1].neuronas
				entrada = []
				for j in range(len(neuronas)):
					entrada.append(neuronas[j].salida)
				capa.calcular_salidas(self.indice_funcion_activacion[i], entrada)

	def realizar_retropropagacion(self, salida, entrada):
		"""
		"""
		for i in range(len(self.red_neuronal)):
			indice = len(self.red_neuronal) - (i+1)
			neuronas_actuales = self.red_neuronal[indice].neuronas
			if indice == len(self.red_neuronal) - 1:
				neuronas_previas = self.red_neuronal[indice - 1].neuronas
				errores = []
				for j in range(len(neuronas_actuales)):
					errores.append(salida - neuronas_actuales[j].salida)
				entrada = []
				for j in range(len(neuronas_previas)):
					entrada.append(neuronas_previas[j].salida)
				self.red_neuronal[indice].calcular_delthas_salida(
					self.indice_funcion_activacion[indice], errores,
					entrada)
			else:
				capa_siguiente = self.red_neuronal[indice + 1]
				if indice == 0:
					self.red_neuronal[indice].calcular_delthas_ocultas(
						self.indice_funcion_activacion[indice],
						capa_siguiente.delthas, entrada,
						neuronas_actuales.neuronas)
				else:
					neuronas_previas = self.red_neuronal[indice - 1].neuronas
					entrada = []
					for j in range(len(neuronas_previas)):
						entrada.append(neuronas_previas[j].salida)
					self.red_neuronal[indice].calcular_delthas_ocultas(
						self.indice_funcion_activacion[indice],
						capa_siguiente.delthas, entrada,
						neuronas_actuales.neuronas)

	def actualizar_parametros_neuronales(self, entrada):
		"""
		"""
		for i in range(len(self.red_neuronal)):
			self.red_neuronal[i].actualizar_biases()
			if i == 0:
				self.red_neuronal[i].actualizar_pesos(entrada)
			else:
				neuronas = self.red_neuronal[i].neuronas
				entrada = []
				for j in range(len(neuronas)):
					entrada.append(neuronas[j].salida)
				self.red_neuronal[i].actualizar_pesos(entrada)