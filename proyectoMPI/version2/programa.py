"""
"""

from perceptron import *
from entrenamiento import *

patrones = [
			[-2.0,[-1.0]],
			[-1.2,[-0.81]],
			[0.4,[-0.31]],
			[0.4,[0.309]],
			[1.2,[0.809]],
			[2.0,[1.0]],
			]
ejemplos = [
			[-1.0],
			[-0.5],
			[0.4],
			[1.0],
			[1.6],
			]

error = 0.0
for i in range(len(patrones)):
	error += patrones[i][1][0]
error /= float(len(patrones))
error *= (15.0/100.0)
if error < 0.0:
	error *= -1.0

print "\nError: %f\n" % error

print
print "Ejemplo: Red Neuronal Perceptron Multicapa"

red = RedNeuronal(1,"RED1","TDA/RED","FNCN/SALIDAOCULTAS",[[2,2],[1]])

red.establecer_valores_alphas(1)

print 
print "+"*57
for i in range(len(red.capas)):
	print "Capa[" + str(i+1) + "]" + "="*50
	for j in range(len(red.capas[i].neuronas)):
		print "Neurona(" + str(j+1) + ")" + "-"*47
		print "Bias: ", red.capas[i].neuronas[j].bias
		print "Alpha: ", red.capas[i].neuronas[j].alpha
		print "Pesos: ", red.capas[i].neuronas[j].pesos
print "+"*57
print

print
algoritmo_retropropagacion(5,error,patrones,red)
print

print 
print "+"*57
for i in range(len(red.capas)):
	print "Capa[" + str(i+1) + "]" + "="*50
	for j in range(len(red.capas[i].neuronas)):
		print "Neurona(" + str(j+1) + ")" + "-"*47
		print "Bias: ", red.capas[i].neuronas[j].bias
		print "Alpha: ", red.capas[i].neuronas[j].alpha
		print "Pesos: ", red.capas[i].neuronas[j].pesos
print "+"*57
print