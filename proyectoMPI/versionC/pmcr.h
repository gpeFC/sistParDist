/******************************************************************************
* Sistemas Paralelos y Distribuidos
* Emanuel GP
* Marzo / 2015
* 
* Proyecto: RNA-Perceptron Multicapa con Retroprogacion.
******************************************************************************/

#include <math.h>

typedef struct Neurona{
	double alpha;
	double salida;
	double bias;
	double** pesos;
}NEURONA;

typedef struct Capa{
	double** delthas;
	NEURONA** neuronas;
}CAPA;

typedef struct Red{
	int** funciones;
	CAPA** capas;
}RED;

double suma_ponderada(int dim, double bias, double entrada[], double pesos[]){
	int i;
	double potencial = 0.0;
	for(i=0; i<dim; i++)
		potencial += (pesos[i] * entrada[i]);
	potencial += bias;
	return potencial;
}

double activacion(int id_fncn, double potencial){
	if(id_fncn == 1)
		return potencial;
	else if(id_fncn == 2)
		return (1.0 / (1.0 + pow(exp(1), -potencial)));
	else if(id_fncn == 3)
		return ((2.0 / (1.0 + pow(exp(1), -potencial))) - 1.0);
	else if(id_fncn == 4)
		return((pow(exp(1), potencial) - pow(exp(1), -potencial)) / (pow(exp(1), potencial) + pow(exp(1), -potencial)));
}

double derivada(int id_fncn, double potencial){
	if(id_fncn == 1)
		return 1.0;
	else if(id_fncn == 2)
		return (activacion(2, potencial)*(1.0 - activacion(2, potencial)));
	else if(id_fncn == 3)
		return ((2.0 * pow(exp(1), -potencial)) / pow(1.0, pow(exp(1), -potencial)));
	else if(id_fncn == 4)
		return (1.0 - pow(activacion(4, potencial), 2.0));
}
