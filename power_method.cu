#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<iostream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#define n 1000

__global__ void Matrix_Product (double *A, double *g, double *C)
// Each thread computes one element of C
// by accumulating results into Cvalue 
{               double Cvalue = 0.00;
                int row = blockIdx.y*blockDim.y+threadIdx.y;
               // int col = blockIdx.x * blockDim.x + threadIdx.x;
        //size of matrix A//
                int N=1000;
                if(row> N ) return;
                for (int e = 0; e < N; e++)
                        {
                        Cvalue += A[N*row+e]*g[e];
                        }
                 C[row]+= Cvalue;                     
}

using namespace std;
int main(){

double a[n*n],x[n],c[n],temp=0,d=2;
 
srand(time(NULL));
for(long int i=0;i<n*n;i++)
{
		a[i]=2*i*314.9568298+100;	
		//cin>>a[i][j];                //generating the matrix a[n][n]
		//cout<<" "<<a[i][j]<<endl;
}
//
for(int i=0;i<n;i++)
{
	x[i]=0.5;
}
x[n-1]=1;

cudaEvent_t start,stop;
        float elapsedTime;    

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0); 

	double *dev_a, *dev_x, *dev_c;
        dim3 griddim(100,1);
        dim3 blockdim(10,1);
	cudaMalloc( (void**)&dev_a, n *n* sizeof(double) );
	 cudaMalloc( (void**)&dev_c, n * sizeof(double) );
        cudaMalloc( (void**)&dev_x, n * sizeof(double) );
	 cudaMemcpy( dev_a,a,n * n * sizeof(double),cudaMemcpyHostToDevice );
	

 while(fabs(d-temp)>0.0000000000001)
    {
		
        for(int i=0;i<n;i++)
        {
            c[i]=0;
	}
           // for(int j=0;j<n;j++)        //portion to be parallelized
	//		{
          //     			 c[i]+=a[i][j]*x[j];
	//		}


  //      cudaMalloc( (void**)&dev_c, n * sizeof(double) );
//        cudaMalloc( (void**)&dev_x, n * sizeof(double) );
//        cudaMalloc( (void**)&dev_a, n *n* sizeof(double) );

        //cudaMemcpy( dev_a,a,n * n * sizeof(double),cudaMemcpyHostToDevice );
        cudaMemcpy( dev_x,x,n * sizeof(double),cudaMemcpyHostToDevice );
        cudaMemcpy( dev_c,c,n * sizeof(double),cudaMemcpyHostToDevice );

        Matrix_Product<<<griddim, blockdim>>>( dev_a, dev_x, dev_c );

        cudaMemcpy( c,dev_c,n * sizeof(double),cudaMemcpyDeviceToHost );

//        cudaFree( dev_a );
  //      cudaFree( dev_x );
    //    cudaFree( dev_c );

        
        for(int i=0;i<n;i++)
		{
        	    x[i]=c[i];
		}
        temp=d;
        d=0;
        
        for(int i=0;i<n;i++)
        {
            if(fabs(x[i])>fabs(d))
                d=x[i];
        }
        for(int i=0;i<n;i++){
            x[i]/=d;
		}
    }
//	 cudaMemcpy( c,dev_c,n * sizeof(double),cudaMemcpyDeviceToHost );


	   cudaFree( dev_a );
        cudaFree( dev_x );
        cudaFree( dev_c );
    
 cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime,start,stop);
       cout<<"\n\nElapsed Time = "<<elapsedTime<<" ms";

    //cout<<d<<endl;
    //for(int i=0;i<n;i++){
	//	cout<<setprecision(30)<<d<<endl;
	//}
//cout<<"Enter the initial guess for eigen vector";
//for(int i=0;i<n;i++){
//	cout<<x[i]<<endl;
//}
//}

return 0;
}
