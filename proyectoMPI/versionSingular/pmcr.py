import math
import random

def pseudoaleatorio(inferior, superior):
	"""
	Regresa un numero real pseudoaleatorio de un rango definido, el numero
	es distinto de las cotas del rango.

	inferior     Cota inferior del rango.
	superior     Cota superior del rango.
	"""
	while True:
		numero = random.uniform(inferior, superior)
		if numero != inferior and numero != superior:
			return numero

def suma_ponderada(bias, entrada, pesos):
	"""
	(Regla de propagacion) Regresa el potencial sinaptico de una neurona.

	bias        Bias/umbral sinaptico de la neurona.
	entrada     Valores presinapticos de la neurona.
	pesos       Pesos sinapticos de la neurona.
	"""
	potencial = 0.0
	for i in range(len(pesos)):
		potencial += (pesos[i] * entrada[i])
	potencial += bias
	return potencial

def activacion(id_funcion, potencial):
	"""
	Regresa la activacion postsinaptica de la neurona.

	id_funcion     Indice que indica la funcion de activacion de la neurona.
	potencial      Potencial sinaptico de la neurona.
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
	if id_funcion == 1:
		return 1.0
	elif id_funcion == 2:
		return (activacion(2, potencial) * (1.0 - activacion(2, potencial)))
	elif id_funcion == 3:
		return ((2.0 * pow(math.e, (-1 * potencial))) / pow(1.0 + pow(math.e, (-1 * potencial)), 2.0))
	elif id_funcion == 4:
		return (1.0 - pow(activacion(4, potencial), 2.0))

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
		self.salida = activacion(id_funcion, suma_ponderada(self.bias, entrada, self.pesos))


class CapaNeuronal(object):
	"""docstring for CapaNeuronal"""
	def __init__(self, total_neurs, total_args, indice_funciones):
		super(CapaNeuronal, self).__init__()
		self.funciones = indice_funciones
		self.delthas = [0.0] * total_neurs
		self.salidas = [0.0] * total_neurs
		self.entrada = []
		self.neuronas = []
		for i in range(total_neurs):
			self.neuronas.append(Neurona(total_args))

	def ajustar_biases(self):
		for i in range(len(self.neuronas)):
			self.neuronas[i].bias += (self.neuronas[i].alpha * self.delthas[i])

	def ajustar_pesos(self):
		for i in range(len(self.neuronas)):
			for j in range(len(self.neuronas[i].pesos)):
				self.neuronas[i].pesos[j] += (self.neuronas[i].alpha * self.delthas[i] * self.entrada[j])

	def calcular_delthas_salida(self, errores):
		for i in range(len(errores)):
			self.delthas[i] = errores[i] * derivada(self.funciones[i],
													suma_ponderada(self.neuronas[i].bias,
													self.entrada, self.neuronas[i].pesos))

	def calcular_delthas_ocultas(self, capa_sig):
		for i in range(len(self.funciones)):
			suma_delthas = 0.0
			for j in range(len(capa_sig.delthas)):
				suma_delthas += (capa_sig.delthas[j] * capa_sig.neuronas[j].pesos[i])
			self.delthas[i] = suma_delthas * derivada(self.funciones[i], suma_ponderada(
													self.neuronas[i].bias, self.entrada,
													self.neuronas[i].pesos))

	def calcular_salidas(self):
		for i in range(len(self.funciones)):
			self.neuronas[i].calcular_salida(self.funciones[i], self.entrada)
			self.salidas[i] = self.neuronas[i].salida


class RedNeuronal(object):
	"""docstring for RedNeuronal"""
	def __init__(self, id_alphas, total_args, indice_funciones):
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
				self.capas[i].entrada = entrada
			else:
				entrada = self.capas[i-1].salidas
				self.capas[i].entrada = entrada
			self.capas[i].calcular_salidas()

	def retropropagacion(self, salida):
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
		for i in range(len(self.capas)):
			self.capas[i].ajustar_biases()
			self.capas[i].ajustar_pesos()

	def imprime_red(self):
		print "="*60
		for i in range(len(self.capas)):
			print "Capa " + str(i+1) + "-"*35
			for j in range(len(self.capas[i].neuronas)):
				print "Neurona " + str(j+1)
				print "\tBias:", self.capas[i].neuronas[j].bias
				print "\tAlpha:", self.capas[i].neuronas[j].alpha
				print "\tPesos:", self.capas[i].neuronas[j].pesos
		print "="*60
		