from funciones import *

class Neurona(object):
	"""docstring for Neurona"""
	def __init__(self, total_args):
		super(Neurona, self).__init__()
		self.alpha = 0.0
		self.salida = 0.0
		self.bias = pseudoaleatorio(-1.0, 1.0)
		self.pesos = []
		for i in range(total_args):
			self.pesos.append(pseudoaleatorio(-1.0,1.0))

	def calcular_salida(self, id_funcion, entrada):
		if id_funcion == 1:
			self.salida = identidad_lineal(suma_ponderada(self.bias, entrada, self.pesos))
		elif id_funcion == 2:
			self.salida = sigmoide_logistico(suma_ponderada(self.bias, entrada, self.pesos))
		elif id_funcion == 3:
			self.salida = sigmoide_tangencial(suma_ponderada(self.bias, entrada, self.pesos))
		elif id_funcion == 4:
			self.salida = sigmoide_hiperbolico(suma_ponderada(self.bias, entrada, self.pesos))


class CapaNeuronal(object):
	"""docstring for CapaNeuronal"""
	def __init__(self, total_neurs, total_args):
		super(CapaNeuronal, self).__init__()
		self.delthas = [0.0] * total_neurs
		self.neuronas = []
		for i in range(total_neurs):
			self.neuronas.append(Neurona(total_args))

	def ajustar_biases(self):
		for i in range(len(self.neuronas)):
			self.neuronas[i].bias += (self.neuronas[i].alpha * self.delthas[i])

	def ajustar_pesos(self, entrada):
		for i in range(len(self.neuronas)):
			for j in range(len(entrada)):
				self.neuronas[i].pesos[j] -= (self.neuronas[i].alpha * self.delthas[i] * entrada[j])

	def calcular_delthas_salida(self, id_funciones, errores, entrada):
		for i in range(len(errores)):
			if id_funciones[i] == 1:
				self.delthas[i] = errores[i] * derivada_lineal(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))
			elif id_funciones[i] == 2:
				self.delthas[i] = errores[i] * derivada_logistica(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))
			elif id_funciones[i] == 3:
				self.delthas[i] = errores[i] * derivada_tangencial(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))
			elif id_funciones[i] == 4:
				self.delthas[i] = errores[i] * derivada_hiperbolica(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos))

	def calcular_delthas_ocultas(self, id_funciones, entrada, capa_sig):
		for i in range(len(self.neuronas)):
			suma_deltha = 0.0
			for j in range(len(capa_sig.neuronas)):
				suma_deltha += (capa_sig.delthas[j] * capa_sig.neuronas[j].pesos[i])
			if id_funciones[i] == 1:
				self.delthas[i] = derivada_lineal(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos)) * suma_deltha
			elif id_funciones[i] == 2:
				self.delthas[i] = derivada_logistica(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos)) * suma_deltha
			elif id_funciones[i] == 3:
				self.delthas[i] = derivada_tangencial(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos)) * suma_deltha
			elif id_funciones[i] == 4:
				self.delthas[i] = derivada_hiperbolica(suma_ponderada(self.neuronas[i].bias, entrada, self.neuronas[i].pesos)) * suma_deltha

	def calcular_salidas(self, id_funciones, entrada):
		for i in range(len(id_funciones)):
			self.neuronas[i].calcular_salida(id_funciones[i], entrada)


class RedNeuronal(object):
	"""docstring for RedNeuronal"""
	def __init__(self, indice_alphas, total_args, indices_fncns):
		super(RedNeuronal, self).__init__()
		self.indice_funciones = indices_fncns
		self.capas = []
		for i in range(len(self.indice_funciones)):
			if i == 0:
				self.capas.append(CapaNeuronal(len(self.indice_funciones[i]), total_args))
			else:
				self.capas.append(CapaNeuronal(len(self.indice_funciones[i]), len(self.indice_funciones[i-1])))
		if indice_alphas == 1:
			alpha = pseudoaleatorio(0.0, 1.0)
			for i in range(len(self.capas)):
				for j in range(len(self.capas[i].neuronas)):
					self.capas[i].neuronas[j].alpha = alpha
		elif indice_alphas == 2:
			for i in range(len(self.capas)):
				alpha = pseudoaleatorio(0.0, 1.0)
				for j in range(len(self.capas[i].neuronas)):
					self.capas[i].neuronas[j].alpha = alpha
		elif indice_alphas == 3:
			for i in range(len(self.capas)):
				for j in range(len(self.capas[i].neuronas)):
					self.capas[i].neuronas[j].alpha = pseudoaleatorio(0.0, 1.0)

	def aplicar_red_neuronal(self, entrada):
		pass

	def realizar_propagacion(self, entrada):
		for i in range(len(self.capas)):
			if i == 0:
				self.capas[i].calcular_salidas(self.indice_funciones[i], entrada)
			else:
				entrada = []
				for j in range(len(self.capas[i-1].neuronas)):
					entrada.append(self.capas[i-1].neuronas[j].salida)
				self.capas[i].calcular_salidas(self.indice_funciones[i], entrada)

	def realizar_retropropagacion(self, salidas, entrada):
		pass

	def ajustar_parametros_neurales(self, entrada):
		for i in range(len(self.capas)):
			self.capas[i].ajustar_biases()
			if i == 0:
				self.capas[i].ajustar_pesos(entrada)
			else:
				entrada = []
				for j in range(len(self.capas[i-1].neuronas)):
					entrada.append(self.capas[i-1].neuronas[j].salida)
				self.capas[i].ajustar_pesos(entrada)