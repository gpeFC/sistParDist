#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv){
	int actual,  total, valor;
	MPI_Status estado;

	MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &total);
		MPI_Comm_rank(MPI_COMM_WORLD, &actual);

		if(actual != 0){
			MPI_Recv(&valor, 1, MPI_INT, (actual-1), 1, MPI_COMM_WORLD, &estado);
			if(actual < total-1)
				MPI_Send(&actual, 1, MPI_INT, (actual+1), 1, MPI_COMM_WORLD);
			printf("Soy el proceso %d y he recibido %d.\n", actual, valor);
		}
		else{
			if(total > 1)
				MPI_Send(&actual, 1, MPI_INT, (actual+1), 1, MPI_COMM_WORLD);
		}

	MPI_Finalize();
}