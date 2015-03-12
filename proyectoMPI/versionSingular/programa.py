from pmcr import *

patrones_and = [
			[0.0, 0.0, [0.0]],
			[0.0, 1.0, [0.0]],
			[1.0, 0.0, [0.0]],
			[1.0, 1.0, [1.0]]
			]

patrones_or = [
			[0.0, 0.0, [0.0]],
			[0.0, 1.0, [1.0]],
			[1.0, 0.0, [1.0]],
			[1.0, 1.0, [1.0]]
			]

patrones_xor = [
			[0.0, 0.0, [0.0]],
			[0.0, 1.0, [1.0]],
			[1.0, 0.0, [1.0]],
			[1.0, 1.0, [0.0]]
			]

patrones_tan = [
			[-2.0,[-1.0]],
			[-1.2,[-0.81]],
			[0.4,[-0.31]],
			[0.4,[0.309]],
			[1.2,[0.809]],
			[2.0,[1.0]],
			]
prueba_tan = [
			[-1.0],
			[-0.5],
			[0.4],
			[1.0],
			[1.6],
			]

error = 0.05

epocas = 1000

indices = [[2,2],[1]]


print
print "Ejemplo: Red Neuronal Perceptron Multicapa"
print 

red = RedNeuronal(1, 1, indices)

red.imprime_red()
print

algoritmo_retropropagacion(epocas, error, patrones_tan, red)

red.imprime_red()
print