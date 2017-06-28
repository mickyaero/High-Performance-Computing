#include <iostream>
#include <stdio.h>

using namespace std;

__host__ __device__ void deriv(double* x0, double* array, double c, int size,  double* ki, double*
        ko);
__global__ void rk4(double* x0, double* array, double* h, int size);

int main(){

    int n_voxel = 10;
    int n_species = 100;
    int species_square = n_species * n_species;
    double x0[n_species];
    double a_mat[species_square];
    double h = 0.005;
    
    double duration = 0.0;
    //Filling up the a_mat and x0
    
    for(int i = 0; i < n_species; ++i){
        x0[i] = i;
        for(int j = 0; j < n_species; ++j){
            a_mat[i * n_species + j] = rand() % 9 + 1; 
        }
    }

    
    //Porting the problem onto gpu
    double *d_x0, *d_a_mat, d_h;
    int  d_n_species;
    
    clock_t start = clock();

    cudaMalloc( (void**)&d_x0, sizeof(double) * n_species );
    cudaMalloc( (void**)&d_a_mat, sizeof(double) * species_square );
    cudaMalloc( (void**)&d_h, sizeof(double) * 1 );
    cudaMalloc( (void**)&d_n_species, sizeof(int) * 1 );

    cudaMemcpy( d_x0, x0, sizeof(double) * n_species, cudaMemcpyHostToDevice );
    cudaMemcpy( d_a_mat, a_mat, sizeof(double) * species_square, cudaMemcpyHostToDevice );
    cudaMemcpy( &d_h, &h, sizeof(double) * 1, cudaMemcpyHostToDevice );
    cudaMemcpy( &d_n_species, &n_species, sizeof(int) * 1, cudaMemcpyHostToDevice );

    dim3 blocks( 1, 1, 1 );
    dim3 threads( 10, 1, 1 );

    rk4 <<< blocks, threads >>> ( d_x0, d_a_mat, &d_h, d_n_species );
    
    cudaMemcpy( x0, d_x0, sizeof(double) * n_species, cudaMemcpyDeviceToHost );
    cudaMemcpy( a_mat, d_a_mat, sizeof(double) * species_square, cudaMemcpyDeviceToHost );
    
    cudaFree( d_x0 );
    cudaFree( d_a_mat );
    cudaFree( &d_h );
    cudaFree( &d_n_species );

    duration = static_cast<double>(clock() - start);
    cout << "Time = " << duration << endl;

/**
    //Running rk4 over all the voxels
    clock_t start = clock();
    for(int i = 0; i < n_voxel; ++i){
        rk4(x0, a_mat, &h, n_species);
       // cout << "x0 address = " << x0 << endl;
       // cout << "x0 value = " << x0[0] << endl;
   }
   duration = static_cast<double>(clock() - start);
   cout << "Time = " << duration << endl;

   **/
}

__host__ __device__ void deriv(double* x0, double* array, double c, int size,  double* ki, double* ko){

    for(int i = 0; i < size; ++i){

        for(int j = 0; j< size; ++j){

            ko[i] = ko[i] + array[ size * i + j ] * (x0[j] + c * ki[j]); 

        }

    }

}

__global__ void rk4(double* x0, double* array, double* h, int size){
    //int** arr = new int*[row];
    //int size = 100;//size of the species
    double* k1_v = new double [size];
    double* k2_v = new double [size];
    double* k3_v = new double [size];
    double* k4_v = new double [size];
    for(int i = 0; i < size; ++i){

        k1_v[i] = x0[i];
        k2_v[i] = x0[i];
        k3_v[i] = x0[i];
        k4_v[i] = x0[i];
    
    }

    deriv(x0, array, 0.0, size, x0, k1_v);
    deriv(x0, array, *h/2.0, size, k1_v, k2_v);
    deriv(x0, array, *h/2.0, size, k2_v, k3_v);
    deriv(x0, array, *h, size, k3_v, k4_v);

    for(int i = 0; i < size; ++i){
        x0[i] = x0[i] + (k1_v[i] + 2.0 *  k2_v[i] + 2.0 * k3_v[i] + k4_v[i]) *
            (*h)/6.0;  
    }
    delete[] k1_v;
    delete[] k2_v;
    delete[] k3_v;
    delete[] k4_v;
   // delete[] arr;
}


