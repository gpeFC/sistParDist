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
	potencial += bias
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
			self.salida = suma_ponderada(self.bias, entrada, self.pesos)
		elif id_funcion == 2:
			self.salida = sigmoide_tangencial(suma_ponderada(self.bias, entrada, self.pesos))


class CapaNeuronal(object):
	"""docstring for CapaNeuronal"""
	def __init__(self, total_neurs, total_args):
		super(CapaNeuronal, self).__init__()
		self.delthas = []
		self.neuronas = []
		for i in range(total_neurs):
			self.neuronas.append(Neurona(total_args))

	def ajustar_biases(self):
		for i in range(len(self.neuronas)):
			self.neuronas[i].bias += (self.neuronas[i].alpha * self.delthas[i])

	def ajustar_pesos(self, entrada):
		for i in range(len(self.neuronas)):
			for j in range(len(self.neuronas[i].pesos)):
				self.neuronas[i].pesos[j] -= (self.neuronas[i].alpha * self.delthas[i] * entrada[j])

	def calcular_delthas_salida(self, id_fncns, errores, entrada):
		for i in range(len(errores)):
			deltha = 0.0
			if id_fncns[i] == 1:
				deltha = errores[i]
				self.delthas.append(errores[i])
			else:
				deltha = errores[i] * derivada_tangencial(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))
			self.delthas.append(deltha)

	def calcular_delthas_ocultas(self, id_fncns, entrada, capa_sig):
		for i in range(len(id_fncns)):
			suma_delthas = 0.0
			for j in range(len(capa_sig.delthas)):
				suma_delthas += (capa_sig.delthas[j] * capa_sig.neuronas[j].pesos[i])
			deltha = 0.0
			if id_fncns[i] == 1:
				deltha = suma_delthas
			else:
				deltha = suma_delthas * derivada_tangencial(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))
			self.delthas.append(deltha)

	def calcular_salidas(self, id_fncns):
		for i in range(len(id_fncns)):
			self.neuronas[i].calcular_salida(id_fncns[i], self.entrada)


class RedNeuronal(object):
	"""docstring for RedNeuronal"""
	def __init__(self, id_alphas, total_args, indice_funciones):
		super(RedNeuronal, self).__init__()
		self.indice_funciones = indice_funciones
		self.capas = []

	def aplicar_red(self):
		pass

	def propagacion(self, entrada):
		pass

	def retropropagacion(self):
		pass

	def ajustar_parametros(self):
		pass
		