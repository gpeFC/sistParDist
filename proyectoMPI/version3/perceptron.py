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
		pass

	def ajustar_pesos(self):
		pass

	def calcular_delthas_salida(self):
		pass

	def calcular_delthas_ocultas(self):
		pass

	def calcular_salidas(self):
		pass