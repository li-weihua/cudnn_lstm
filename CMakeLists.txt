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
set(CUDNN_INCLUDE_DIR "${CUDNN_INCLUDE}")

# extract cudnn version
if(CUDNN_INCLUDE_DIR)
  if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_H_CONTENTS)
  else()
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn.h" CUDNN_H_CONTENTS)
  endif()

  string(REGEX MATCH "define CUDNN_MAJOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_VERSION_MAJOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
  string(REGEX MATCH "define CUDNN_MINOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_VERSION_MINOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_VERSION_PATCH ${CMAKE_MATCH_1} CACHE INTERNAL "")

  set(CUDNN_VERSION
    "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}"
    CACHE
    STRING
    "cuDNN version"
  )
  unset(CUDNN_H_CONTENTS)
endif()

message("CUDNN_VERSION: ${CUDNN_VERSION}")

if (CUDNN_VERSION GREATER_EQUAL "8.0") 
    set(CUDNN_INFER_LIBS "${CUDNN_ROOT}/lib64/libcudnn_ops_infer.so" "${CUDNN_ROOT}/lib64/libcudnn_adv_infer.so")
else()
    set(CUDNN_INFER_LIBS "${CUDNN_ROOT}/lib64/libcudnn.so")
endif()


add_executable(run lstm.cc)
target_include_directories(run PRIVATE ${CUDNN_INCLUDE})
target_link_libraries(run PRIVATE ${CUDNN_INFER_LIBS} CUDA::cudart)

if (CUDNN_VERSION GREATER_EQUAL "8.0") 
    add_executable(run_v8 lstm_v8.cc)
    target_include_directories(run_v8 PRIVATE ${CUDNN_INCLUDE})
    target_link_libraries(run_v8 PRIVATE ${CUDNN_INFER_LIBS} CUDA::cudart)
endif()
