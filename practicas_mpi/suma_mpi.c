#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv){
	int mynode, totalnodes;
	int suma = 0, startval, endval, accum, i;
	MPI_Status status;

	MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
		MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

		startval = 1000 * mynode / totalnodes+1;
		endval = 1000 * (mynode+1) / totalnodes;

		for(i=startval; i<=endval; i++)
			suma += i;

		if(mynode != 0){
			MPI_Send(&suma, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
			printf("La suma de %d hasta %d es: %d\n", startval, endval, suma);
		}
		else{
			for(i=1;i<totalnodes;i++){
				MPI_Recv(&accum, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
				suma += accum;
			}
			if(totalnodes > 1)
				printf("La suma de %d hasta %d es: %d\n", startval, endval, accum);
		}
		if(mynode == 0)
			printf("La suma total es: %d\n", suma);
	MPI_Finalize();
}