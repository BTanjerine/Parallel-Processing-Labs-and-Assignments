// CS3700 Example matrix multpilcation using GPU

#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define TILE_WIDTH 2
#define WIDTH 6

// Kernel function execute by the device (GPU)
__global__ void product (float *d_a, float *d_b, float *d_c, const int n) {
   int col = blockIdx.x * blockDim.x + threadIdx.x ;
   int row = blockIdx.y * blockDim.y + threadIdx.y ;

   float sum = 0;
   if (row < n && col < n) {
      for (int i = 0 ; i<n ; ++i) {
         sum += d_a[row * n + i ] * d_b[i * n + col] ;
      }
      d_c[row * n + col] = sum;
   }
}

// Kernel for finding the sum of 2 matrices
__global__ void sum_m (float *d_a, float *d_b, float *d_c, const int n) {
   int col = blockIdx.x * blockDim.x + threadIdx.x ;
   int row = blockIdx.y * blockDim.y + threadIdx.y ;

   float sum = 0;
   if(row < n && col < n){
      //add corresponding indices together
      sum = d_a[row*n+col] + d_b[row*n+col]; 

      //set that value to that square
      d_c[row * n + col] = sum; 

      //debug print location and result	
      //printf("d_a[%d] = %1.1f, d_b[%d] = %1.1f, sum[%d] = %1.1f \r\n", row*n+col, d_b[row*n+col], row*n+col, d_a[row*n+col], row*n+col, d_c[row*n+col]);
   }
   
}

// Utility function to print the input matrix
void printMatrix (float m[][WIDTH]) {
   int i, j;
   for (i = 0; i<WIDTH; ++i) {
      for (j = 0; j< WIDTH; ++j) {
         printf ("%d\t", (int)m[i][j]);
      }
      printf ("\n");
   }
}

// Main function execute by the host (CPU)
int main () {
   // host matrices
   float host_a[WIDTH][WIDTH],
         host_b[WIDTH][WIDTH],
         host_c[WIDTH][WIDTH];

   // device arrays
   float *device_a, *device_b, *device_c;

   int i, j;

   // initialize host matrices using random numbers
   time_t t;
   srand ((unsigned) time(&t));

   for (i = 0; i<WIDTH; ++i) {
      for (j = 0; j<WIDTH; j++) {
         host_a[i][j] = (float) (rand() % 50);
         host_b[i][j] = (float) (rand() % 50);
      }
   }

   printf ("Matrix A:\n");
   printMatrix (host_a);
   printf ("\n");

   printf ("Matrix B:\n");
   printMatrix (host_b);
   printf ("\n");

   // allocate device memory for input matrices
   size_t deviceSize = WIDTH * WIDTH * sizeof (float);
   cudaMalloc ((void **) &device_a, deviceSize);
   cudaMalloc ((void **) &device_b, deviceSize);

   // copy host matrices to device
   cudaMemcpy (device_a, host_a, deviceSize, cudaMemcpyHostToDevice );
   cudaMemcpy (device_b, host_b, deviceSize, cudaMemcpyHostToDevice );

   // allocate device memory to store computed result
   cudaMalloc((void **) &device_c, deviceSize) ;

   /*
    * #define TILE_WIDTH 2
    * #define WIDTH 6
    */
   dim3 dimGrid (WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH);
   dim3 dimBlock (WIDTH, WIDTH);

   /****************** PRODUCT *********************/
   product<<<dimGrid, dimBlock>>> (device_a, device_b, device_c, WIDTH);
   // copy result from device back to host
   cudaMemcpy (host_c, device_c, deviceSize, cudaMemcpyDeviceToHost);

   // output the computed result matrix
   printf ("A x B: \n");
   printMatrix (host_c);

   /****************** SUM  *********************/
   sum_m<<<dimGrid, dimBlock>>>(device_a, device_b, device_c, WIDTH);

   // copy result from device back to host
   cudaMemcpy (host_c, device_c, deviceSize, cudaMemcpyDeviceToHost);

   // output the computed result matrix
   printf ("A + B: \n");
   printMatrix (host_c);

   cudaFree (device_a);
   cudaFree (device_b);
   cudaFree (device_c);
   return 0;
}
