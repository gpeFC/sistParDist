import math
import random

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
		self.delthas = []
		for i in range(len(errores)):
			deltha = 0.0
			if id_fncns[i] == 1:
				deltha = errores[i]
			else:
				deltha = errores[i] * derivada_tangencial(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))
			self.delthas.append(deltha)

	def calcular_delthas_ocultas(self, id_fncns, entrada, capa_sig):
		self.delthas = []
		for i in range(len(id_fncns)):
			suma_delthas = 0.0
			for j in range(len(capa_sig.delthas)):
				suma_delthas += (capa_sig.delthas[j] * capa_sig.neuronas[j].pesos[i])
			#print "SumaDeltha:", suma_delthas
			deltha = 0.0
			if id_fncns[i] == 1:
				deltha = suma_delthas
			else:
				deltha = suma_delthas * derivada_tangencial(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))
			self.delthas.append(deltha)

	def calcular_salidas(self, id_fncns, entrada):
		for i in range(len(id_fncns)):
			self.neuronas[i].calcular_salida(id_fncns[i], entrada)


class RedNeuronal(object):
	"""docstring for RedNeuronal"""
	def __init__(self, id_alphas, total_args, indice_funciones):
		super(RedNeuronal, self).__init__()
		self.indice_funciones = indice_funciones
		self.capas = []
		for i in range(len(self.indice_funciones)):
			if i == 0:
				self.capas.append(CapaNeuronal(len(self.indice_funciones[i]), total_args))
			else:
				self.capas.append(CapaNeuronal(len(self.indice_funciones[i]), len(self.indice_funciones[i-1])))
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
		salidas = []
		for i in range(len(entradas)):
			self.propagacion(entradas[i])
			salida = []
			for j in range(len(self.capas[-1].neuronas)):
				salida.append(self.capas[-1].neuronas[j].salida)
			salidas.append(salida)
		return salidas

	def propagacion(self, entrada):
		for i in range(len(self.capas)):
			if i == 0:
				self.capas[i].calcular_salidas(self.indice_funciones[i], entrada)
			else:
				entrada2 = []
				for j in range(len(self.capas[i-1].neuronas)):
					entrada2.append(self.capas[i-1].neuronas[j].salida)
				self.capas[i].calcular_salidas(self.indice_funciones[i], entrada2)

	def retropropagacion(self, salida, entrada):
		lista = range(len(self.capas))
		for i in lista.__reversed__():
			if i == len(self.capas) - 1:
				errores = []
				for j in range(len(self.capas[i].neuronas)):
					errores.append(salida[j] - self.capas[i].neuronas[j].salida)
				entrada2 = []
				for j in range(len(self.capas[i-1].neuronas)):
					entrada2.append(self.capas[i-1].neuronas[j].salida)
				self.capas[i].calcular_delthas_salida(self.indice_funciones[i], errores, entrada2)
			else:
				capa_siguiente = self.capas[i+1]
				if i == 0:
					self.capas[i].calcular_delthas_ocultas(self.indice_funciones[i], entrada, capa_siguiente)
				else:
					entrada2 = []
					for j in range(len(self.capas[i-1].neuronas)):
						entrada2.append(self.capas[i-1].neuronas[j].salida)
					self.capas[i].calcular_delthas_ocultas(self.indice_funciones[i], entrada2, capa_siguiente)

	def ajustar_parametros(self, entrada):
		for i in range(len(self.capas)):
			#print "Delthas(%d):" % (i+1), self.capas[i].delthas
			self.capas[i].ajustar_biases()
			if i == 0:
				self.capas[i].ajustar_pesos(entrada)
			else:
				entrada2 = []
				for j in range(len(self.capas[i-1].neuronas)):
					entrada2.append(self.capas[i-1].neuronas[j].salida)
				self.capas[i].ajustar_pesos(entrada2)
		