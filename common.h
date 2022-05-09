#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>
#include <cudnn.h>

// cuda check macro
#define CUDA_CHECK(statments)                                                  \
  do {                                                                         \
    cudaError_t status = (statments);                                          \
    if (cudaSuccess != status) {                                               \
      std::fprintf(stderr, "[%s:%d] cuda error: %s, %s\n", __FILE__, __LINE__, \
                   cudaGetErrorName(status), cudaGetErrorString(status));      \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

// cudnn check macro
#define CUDNN_CHECK(statments)                                                                 \
  do {                                                                                         \
    cudnnStatus_t status = (statments);                                                        \
    if (CUDNN_STATUS_SUCCESS != status) {                                                      \
      std::fprintf(stderr, "[%s:%d] cudnn code: %d, detail: %s\n", __FILE__, __LINE__, status, \
                   cudnnGetErrorString(status));                                               \
      std::abort();                                                                            \
    }                                                                                          \
  } while (false)
