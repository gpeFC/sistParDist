from perceptron import *
from entrenamiento import *

print "\nRed Multicapa\n"

patrones = [
			[-2.0,[-1.0]],
			[-1.2,[-0.81]],
			[0.4,[-0.31]],
			[0.4,[0.309]],
			[1.2,[0.809]],
			[2.0,[1.0]]
			]

ejemplos = [[-1.5],[-0.8],[0.4],[0.9],[1.6]]

epocas = 30

error = 0.0005

indices = [[2,2],[1]]

bias = pseudoaleatorio(-1.0,1.0)

pesos = [pseudoaleatorio(-1.0,1.0)]

entrada = [-2.0]

potencial = suma_ponderada(bias,entrada,pesos)

activacion = sigmoide_tangencial(potencial)

print "\tDatos\n"
print "Bias:", bias
print "Entrada:", entrada
print "Pesos:", pesos
print "Potencial:", potencial
print "Activacion:", activacion
print 


"""
red = RedNeuronal(1,"RED","TDA/RED","FNCN/CAPA",indices)

red.establecer_valores_alphas(1)

imprime_red(red)

algoritmo_retropropagacion(epocas,error,patrones,red)

print 

imprime_red(red)
"""