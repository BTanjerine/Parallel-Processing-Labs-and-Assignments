#include "mpi.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
 
#define MAX_POINTS 1000000

using namespace std;

int main (int argc, char* argv[]){
	int n_id, num_proc;
	int namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	double start_time, run_time;

	//start mpi
	MPI_Init(&argc, &argv);

	//give us data about node (total, rank, name)
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &n_id);
	MPI_Get_processor_name(processor_name, &namelen);
	
	printf("process %d on %s\r\n", n_id, processor_name);

	//seed random number gen with time and rank	
	srand(time(NULL)+n_id);

	//find the amount of data each node is processing 
	int node_data_len = (int)(floor(MAX_POINTS/num_proc));
	
	printf("data length: %d\n", node_data_len);	

	//total points that are in the circle
	int total_points;
    	int max_points = MAX_POINTS;
    

    	if(n_id == 0){
	    	//master work
		    start_time = MPI_Wtime(); //start timer

		    for(int i=0; i<node_data_len; i++){
			    double x, y, dist;

			    x = (double)(rand())/(double)(RAND_MAX);
			    y = (double)(rand())/(double)(RAND_MAX);
			
	    		dist = (x*x) + (y*y);

			    if(dist <= 1) total_points++;
    		}
    	}
    	else{
	    	//slave work
		    for(int i=0; i<node_data_len; i++){
			    double x, y, dist;

		    	x = (double)(rand())/(double)(RAND_MAX);
			    y = (double)(rand())/(double)(RAND_MAX);
			
		    	dist = (x*x) + (y*y);

			    if(dist <= 1) total_points++;
		    }
	    }

	    //use a reducer to find the sum
	    int sum_points = 0; 
	    MPI_Reduce(&total_points, &sum_points, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	    if(n_id == 0){
		    printf("sum points: %d\n", sum_points);	

		    //estimate pi
		    double mpi_pi = 4.0*((double)(sum_points)/(double)(max_points));
		    //get total run time	
		    run_time = MPI_Wtime() - start_time;

		    printf("run time: %1.6f Estimate PI: %1.5f DELTA: %1.4f\n",run_time, mpi_pi, M_PI-mpi_pi);
	    }

	MPI_Finalize();		
}


