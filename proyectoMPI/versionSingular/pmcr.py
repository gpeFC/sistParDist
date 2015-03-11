import math
import random

def pseudoaleatorio(inferior, superior):
	while True:
		numero = random.uniform(inferior, superior)
		if numero != inferior and numero != superior:
			return numero

def suma_ponderada(bias, entrada, pesos):
	potencial = 0.0
	for i in range(len(pesos)):
		potencial += (pesos[i] * entrada[i])
	return potencial

def sigmoide_tangencial(potencial):
	return ((2.0 / (1.0 + pow(math.e, (-1.0 * potencial)))) - 1.0)

def derivada_tangencial(potencial):
	return ((2.0 * pow(math.e, (-1 * potencial))) / pow(1.0 + pow(math.e, (-1 * potencial)), 2.0))

class Neurona(object):
	"""docstring for Neurona"""
	def __init__(self, total_args):
		super(Neurona, self).__init__()
		self.alpha = 0.0
		self.salida = 0.0
		self.bias = pseudoaleatorio(-1.0, 1.0)
		self.pesos = []
		for i in range(total_args):
			self.pesos.append(pseudoaleatorio(-1.0, 1.0))

	def calcular_salida(self, id_funcion, entrada):
		if id_funcion == 1:
			self.salida = suma_ponderada(self.bias,
											entrada, self.pesos)
		elif id_funcion == 2:
			self.salida = sigmoide_tangencial(suma_ponderada(self.bias,
											entrada, self.pesos))

class RedNeuronal(object):
			"""docstring for RedNeuronal"""
			def __init__(self, indice_alphas, total_args, indice_fncns):
				super(RedNeuronal, self).__init__()
				self.indice_funciones = indice_fncns
				self.delthas = []
				self.capas_neurales = []
				for i in range(len(self.indice_funciones)):
					capa = []
					for j in range(len(self.indice_funciones[i])):
						if i == 0:
							capa.append(Neurona(total_args))
						else:
							capa.append(Neurona(len(self.indice_funciones[i-1])))
					self.capas_neurales.append(capa)
				if indice_alphas == 1:
					alpha = pseudoaleatorio(0.0, 1.0)
					for i in range(len(self.capas_neurales)):
						for j in range(len(self.capas_neurales[i])):
							self.capas_neurales[i][j].alpha = alpha
				elif indice_alphas == 2:
					for i in range(len(self.capas_neurales)):
						alpha = pseudoaleatorio(0.0, 1.0)
						for j in range(len(self.capas_neurales[i])):
							self.capas_neurales[i][j].alpha = alpha
				elif indice_alphas == 3:
					for i in range(len(self.capas_neurales)):
						for j in range(len(self.capas_neurales[i])):
							self.capas_neurales[i][j].alpha = pseudoaleatorio(0.0, 1.0)

			def aplicar_red(self):
				pass

			def propagacion(self, entrada):
				for i in range(len(self.capas_neurales)):
					for j in range(len(self.capas_neurales[i])):
						if i == 0:
							self.capas_neurales[i][j].calcular_salida(self.indice_funciones[i][j], entrada)
						else:
							entrada = []
							for k in range(len(self.capas_neurales[i-1])):
								entrada.append(self.capas_neurales[i-1][k].salida)
							self.capas_neurales[i][j].calcular_salida(self.indice_funciones[i][j], entrada)

			def retropropagacion(self, salidas, entrada):
				for i in self.capas_neurales.__reversed__():
					delthas = []
					if i == len(self.capas_neurales) - 1:
						errores = []
						for j in range(len(self.capas_neurales[i])):
							errores.append((salidas[j] - self.capas_neurales[i][j].salida))
						entrada = []
						for j in range(len(self.capas_neurales[i-1])):
							entrada.append(self.capas_neurales[i-1][j].salida)
						for j in range(len(self.capas_neurales[i])):
							if self.indice_funciones[i][j] == 1:
								delthas.append(errores[j])
							else:
								delthas.append((derivada_tangencial(suma_ponderada(self.capas_neurales[i][j].bias, entrada, self.capas_neurales[i][j].pesos))*errores[j]))
						self.delthas.append(delthas)
					else:
						if i == 0:
							pass
						self.delthas.append(delthas)

			def ajustar_parametros(self):
				pass

