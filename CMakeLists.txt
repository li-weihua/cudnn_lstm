cmake_minimum_required(VERSION 3.18)

project(LSTM CXX)

# set use c++ 17
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF) # do not use gnu extension
endif()

# set default build type
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

if (NOT CUDNN_ROOT)
    #set(CUDNN_ROOT ${CUDAToolkit_LIBRARY_ROOT})
    message(FATAL_ERROR "should ste CUDNN_ROOT by -DCUDNN_ROOT=/path/to/cudnn/")
endif()

message("cudnn root: ${CUDNN_ROOT}")

set(CUDNN_INCLUDE "${CUDNN_ROOT}/include")
set(CUDNN_INFER_LIBS "${CUDNN_ROOT}/lib64/libcudnn_ops_infer.so" "${CUDNN_ROOT}/lib64/libcudnn_adv_infer.so")

add_executable(run lstm.cc)
target_include_directories(run PRIVATE ${CUDNN_INCLUDE})
target_link_libraries(run PRIVATE ${CUDNN_INFER_LIBS} CUDA::cudart)
