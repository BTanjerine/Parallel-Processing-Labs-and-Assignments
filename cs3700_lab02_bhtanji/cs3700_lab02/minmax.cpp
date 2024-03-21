#include "mpi.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits.h>

#define ARRAY_SIZE 1000000

int main (int argc,  char *argv[]) {

   int myid, numprocs;
   int namelen;
   int* numbers = new int[ARRAY_SIZE];
   char processor_name[MPI_MAX_PROCESSOR_NAME];

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Get_processor_name(processor_name, &namelen);
 
   std::srand(std::time(NULL)+myid); // seed random number based on time and rank 

   printf("Process %d on %s\n", myid, processor_name);
 
   for (int i=0; i<ARRAY_SIZE; i++)
      numbers[i] = std::rand();  //could be randomly generated

   int s = (int)floor(ARRAY_SIZE/numprocs);
   int s0 = s + ARRAY_SIZE%numprocs;

   int startIndex = s0 + (myid-1)*s;
   int endIndex = startIndex + s;

   double startwtime;
   if (myid == 0) {
      startwtime = MPI_Wtime();
   }

   int i;
   int loc_min = INT_MAX, loc_max = 0;
   
   if (myid == 0) {
      // master worker - comput the master's numbers
      for (i=0; i<s0; i++) {
         if(numbers[i] < loc_min)
            loc_min = numbers[i];         

         if(numbers[i] > loc_max)
            loc_max = numbers[i];         
      }
      printf("Process %d - startIndex 0 endIndex %d; local min %ld; local max %ld\n",
             myid, s0-1,loc_min, loc_max);
   } else {
      //slave's work
      for (i= startIndex; i<endIndex; i++) {
         if(numbers[i] < loc_min)
            loc_min = numbers[i];         

         if(numbers[i] > loc_max)
            loc_max = numbers[i];         
      }
      printf("Process %d - startIndex 0 endIndex %d; local min %ld; local max %ld\n",
             myid, s0-1,loc_min, loc_max);
   }

   int min = 0, max = 0;
   MPI_Reduce(&loc_max, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&loc_min, &min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

   if (myid == 0) {
      double runTime = MPI_Wtime() - startwtime;
      printf("Execution time (sec) = %f min = %ld max = %ld \n",
             runTime, min, max);
   }

   delete[] numbers;

   MPI_Finalize();
}

