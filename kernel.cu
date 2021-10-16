

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cooperative_groups.h"

#include <stdio.h>
#include <iostream>

#define CHUNKSIZE 6 // number of vectors of B residing in local shared memory
#define PDIM 3      // vector space dimension

cudaError_t addWithCuda(const int *a, const int *b, float *c, unsigned int BPG, unsigned int TPB, unsigned int size);


__device__ float euclidean(int* a, int* b)
{
    float d = 0.0;
    for (int idx = 0; idx < PDIM; idx++)
    {
        d += ((a[idx] - b[idx]) * (a[idx] - b[idx]));
    }
    return sqrtf(d);
}


// A and B are arrays holding integer coordinate vectors of dimensionality PDIM
// layout e.g. A[v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2], ...]
__global__ void hDist(const int* A, const int* B, float* dists, const int cardA, const int cardB)
{
    // shared cache for CHUNKSIZE point vectors of input matrix B
    // shared memory is accessible on a per-block basis
    __shared__ int chunk_cache_B[CHUNKSIZE * PDIM];

    int trdix = threadIdx.x;
    int blkix = blockIdx.x;

    // populate block-local shared cache with vectors from B
    for (int vecB_ix = 0; vecB_ix < CHUNKSIZE; vecB_ix++) {
        for (int dim_idx = 0; dim_idx < PDIM; dim_idx++) {
            chunk_cache_B[(vecB_ix * PDIM) + dim_idx + trdix] = B[((blkix * CHUNKSIZE) + vecB_ix) * PDIM + dim_idx + trdix];
        }
    }

    __syncthreads();

    int vecA_ix = 0;
    int vector_cache_A[PDIM] = { 0 };
    float dist_cache[CHUNKSIZE] = { 0.0 };

    while ((blkix * CHUNKSIZE) < cardB) {

        while (vecA_ix < cardA) {
            for (int dim_idx = 0; dim_idx < PDIM; dim_idx++) {
                vector_cache_A[dim_idx] = A[vecA_ix * dim_idx];
            }
            
            dist_cache[trdix] = euclidean(vector_cache_A, &chunk_cache_B[trdix * PDIM]);

            vecA_ix += 1;
        }
        dists[blkix] = 1234;
    }
    // dists[trix] = sqrtf( (A[trix] + B[trix]) * (A[trix] + B[trix]));
}



int main()
{
    const int pointdim = 3;
    const int ca = 3;
    const int cb = 4;

    const int a[ca * pointdim] = { 1,2,3, 4,5,6, 7,8,9 };
    const int b[cb * pointdim] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 };
    float dsts[ca * cb] = { 0.0 };


    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(a, b, dsts, 2, 6, ca * cb);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    for (int i = 0; i < ca*cb; i++) {
        std::cout << dsts[i] << std::endl;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(const int* a, const int* b, float* c, unsigned int BPG, unsigned int TPB, unsigned int size)
{


    std::cout << "Got size: " << size << std::endl;

    int *dev_a = 0;
    int *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;

    if ((BPG * TPB) != size) {
        fprintf(stderr, "INVALID BPG TPB");
        cudaStatus = cudaSetDevice(0);
        return cudaStatus;
    }


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    hDist<<<BPG, TPB>>>(dev_a, dev_b, dev_c, 1, 1);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
