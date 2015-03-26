#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){
	int myid, numprocs;
	FILE *arch;
	MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
		MPI_Comm_rank(MPI_COMM_WORLD, &myid);
		printf("Soy el procesador %d de un total de %d\n", myid, numprocs);
		if(myid == 1){
			arch = fopen("//home/emanuelgp/repositorios/sistParDist/practicas_mpi/archivo.txt", "w");
			fprintf(arch, "Hola soy el procesador 1\n");
			fclose(arch);
		}
	MPI_Finalize();
	return 0;
}