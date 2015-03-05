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
