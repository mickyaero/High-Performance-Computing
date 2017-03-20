//CUDA code for matrix multiplicationn . The values of a,b,c,q have to changed according to N
#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>
__global__ void Product (float *a, float *b, float *c)
{
// Out of all the threads created each one computes 1 value of C and stores into cval

float cval = 0.00;
int R = blockIdx.y * blockDim.y + threadIdx.y; //Row of the matrix
int C = blockIdx.x * blockDim.x + threadIdx.x; //Column of the matrix
//Defining the size of the matrix//
int N=1000;
if(R> N || C > N ){
    return;
}
for (int j = 0; j < N; j++)
{
cval += a[R * N+ j] *b[j * N + C];
			
}
c[R * N + C]+= cval;                     
}                       
                       
  using namespace std;                

int main(){

//The timing function         
cudaEvent_t start,stop;
float time;	
int N=5000;	
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);	

static float a[25000000],b[25000000];
static float c[25000000];



//Inputting values in the matrix
        
long int  q = 25000000;   // Standard int runs out of memory so long int used
int i=0;
//For checking the matrix multiplication all entries are 1
 while(i != q)
 {
   a[i] = 1;
   b[i] = 1;
  
   i++;
                }

int o=0;
//for(int m=0;m<N;m++){
//for(int n=0;n<N;n++){
//a[o]=m+n;
//b[o]=m*n;
//o=o+1;
//}}
//This section is the GPU part

        float *device_a, *device_b, *device_c;
	dim3 griddimension(500,500); // The dimension of the total grid (Blocks)
	dim3 blockdimension(10,10);  // The dimension of one block ( threads in one block)

//Allocating memory in the device for the matrices: device_a,b,c are device variables
cudaMalloc( (void**)&device_c, q * sizeof(float) );
cudaMalloc( (void**)&device_b, q * sizeof(float) );
cudaMalloc( (void**)&device_a, q * sizeof(float) );
//Copying the variables from CPU to GPU
cudaMemcpy( device_a,a,q * sizeof(float),cudaMemcpyHostToDevice );
cudaMemcpy( device_b,b,q * sizeof(float),cudaMemcpyHostToDevice );
cudaMemcpy( device_c,c,q * sizeof(float),cudaMemcpyHostToDevice );

Product<<<griddimension, blockdimension>>>( device_a, device_b, device_c ); //The device function Product is called

cudaMemcpy( c,device_c,q * sizeof(float),cudaMemcpyDeviceToHost );
cudaFree( device_a );
cudaFree( device_b );
cudaFree( device_c );

cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

cudaEventElapsedTime(&time,start,stop);
cout<<"\n\nTime = "<<time<<" ms";

//For printing the matrix
long int g=N*N,d=0;
while(d!=g){
printf("%f\n",c[d]);
d=d+1;
}
//}


}
              
