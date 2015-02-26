from perceptron import *
from entrenamiento import *


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



print "\nRed Multicapa\n"

patrones = [
			[-2.0,[-1.0]],
			[-1.2,[-0.81]],
			[0.4,[-0.31]],
			[0.4,[0.309]],
			[1.2,[0.809]],
			[2.0,[1.0]]
			]

ejemplos = [
			[-1.5],
			[-0.8],
			[0.4],
			[0.9],
			[1.6]
			]

epocas = 30

error = 0.0005

indices = [[2,2],[1]]

red = RedNeuronal(1,"RED","TDA/RED","FNCN/CAPA",indices)

red.establecer_valores_alphas(1)

imprime_red(red)

algoritmo_retropropagacion(epocas,error,patrones,red)

print 

imprime_red(red)