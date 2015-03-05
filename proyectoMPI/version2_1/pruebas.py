from funciones import *
from perceptron import *
from random import shuffle

def imprime_datos(red):
	for i in range(len(red.capas)):
		print "-"*30
		for j in range(len(red.capas[i].neuronas)):
			print "Neurona:", j+1
			print "  Alpha:", red.capas[i].neuronas[j].alpha
			print "  Bias:", red.capas[i].neuronas[j].bias
			print "  Pesos:", red.capas[i].neuronas[j].pesos
			print "  Salida:", red.capas[i].neuronas[j].salida

pts_ent = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
pts_sld = [[0.0],[1.0],[1.0],[1.0]]

red = RedNeuronal(1,2,[[2,2],[1]])

print
imprime_datos(red)
print

lista = range(len(pts_ent))

for i in range(10):
	print "Epoca:", i+1
	errores = 0
	shuffle(lista)
	for j in lista:
		red.realizar_propagacion(pts_ent[j])
		error = 0.0
		for k in range(len(red.capas[-1].neuronas)):
			error += red.capas[-1].neuronas[k].salida
		error /= float(len(red.capas[-1].neuronas))
		error = pts_sld[j][0] - error
		for k in range(len(red.capas)):
			for l in range(len(red.capas[k].delthas)):
				red.capas[k].delthas[l] = error
		if error != 0.0:
			errores += 1
			red.ajustar_parametros_neurales(pts_ent[j])
	for j in range(len(red.capas)):
		for k in range(len(red.capas[j].neuronas)):
			print "Pesos(%d)(%d):" % (j+1,k+1), red.capas[j].neuronas[k].pesos
	if errores == 0:
		break

print
imprime_datos(red)
print