#ifndef CUDA_STREAM_MANAGER_H
#define CUDA_STREAM_MANAGER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h> 


struct CudaStreamManager {
    const size_t num_expert;
    cublasHandle_t* handles;
    cudaStream_t* streams;

    CudaStreamManager(const size_t num_expert_) : num_expert(num_expert_) {
        streams = new cudaStream_t[num_expert];
		handles = new cublasHandle_t[num_expert];
        for (size_t i=0; i<num_expert; ++i) {
			checkCudaErrors(cublasCreate(handles + i));
			checkCudaErrors(cudaStreamCreate(streams + i));
			checkCudaErrors(cublasSetStream(handles[i], streams[i]));
		}
    }

    ~CudaStreamManager() {
        for (size_t i=0; i<num_expert; ++i) {
            checkCudaErrors(cudaStreamDestroy(streams[i]));
			checkCudaErrors(cublasDestroy(handles[i]));
		}
    }
}; 

CudaStreamManager* getCudaStreamManager(const size_t num_expert);

#endif  // CUDA_STREAM_MANAGER 
