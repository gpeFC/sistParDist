from funciones import *

class Neurona(object):
	"""docstring for Neurona"""
	def __init__(self,total_args):
		super(Neurona, self).__init__()
		self.alpha = 0.0
		self.salida = 0.0
		self.bias = pseudoaleatorio(-1.0,1.0)
		self.pesos = []
		for i in range(total_args):
			self.pesos.append(pseudoaleatorio(-1.0,1.0))

	def calcular_salida(self,id_funcion,entrada):
		if id_funcion == 1:
			self.salida = identidad_lineal(suma_ponderada(self.bias,
											entrada,self.pesos))
		elif id_funcion == 2:
			self.salida = sigmoide_logistico(suma_ponderada(self.bias,
											entrada,self.pesos))
		elif id_funcion == 3:
			self.salida = sigmoide_tangencial(suma_ponderada(self.bias,
											entrada,self.pesos))
		elif id_funcion == 4:
			self.salida = sigmoide_hiperbolico(suma_ponderada(self.bias,
											entrada,self.pesos))

class CapaNeuronal(object):
	"""docstring for CapaNeuronal"""
	def __init__(self,total_neurs,total_args,indices_funciones):
		super(CapaNeuronal, self).__init__()
		self.indices_funciones = indices_funciones
		self.delthas = [0.0]*total_neurs
		self.salidas = [0.0]*total_neurs
		self.entradas = [0.0]*total_args
		self.neuronas = []
		for i in range(total_neurs):
			self.neuronas.append(Neurona(total_args))

	def ajustar_biases(self):
		for i in range(len(self.neuronas)):
			self.neuronas[i].bias += (self.neuronas[i].alpha * self.delthas[i])

	def ajustar_pesos(self):
		for i in range(len(self.neuronas)):
			for j in range(len(self.entradas)):
				self.neuronas[i].pesos[j] -= (self.neuronas[i].alpha * self.delthas[i] * self.entradas[j])

	def calcular_delthas_salida(self, errores):
		for i in range(len(errores)):
			if id_funciones[i] == 1:
				self.delthas[i] = errores[i] * derivada_lineal(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))
			elif id_funciones[i] == 2:
				self.delthas[i] = errores[i] * derivada_logistica(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))
			elif id_funciones[i] == 3:
				self.delthas[i] = errores[i] * derivada_tangencial(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))
			elif id_funciones[i] == 4:
				self.delthas[i] = errores[i] * derivada_hiperbolica(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))

	def calcular_delthas_ocultas(self):
		pass

	def calcular_salidas(self):
		pass