#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "common.h"

namespace util {

template <typename T>
std::vector<T> ReadRawDataToHost(const std::string &file_name) {
  std::ifstream fp(file_name, std::ios::binary | std::ios::ate);
  if (!fp.is_open()) {
    std::fprintf(stderr, "cannot open file: %s\n", file_name.c_str());
    std::abort();
  }

  size_t size = fp.tellg();
  fp.seekg(0, std::ios::beg);

  int n = size / sizeof(T);

  std::vector<T> v(n);

  fp.read(reinterpret_cast<char *>(v.data()), size);
  return v;
}

template <typename T>
void ReadRawDataToGPU(const std::string &file_name, T *d_data) {
  auto h_data = ReadRawDataToHost<T>(file_name);
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), sizeof(T) * h_data.size(), cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> SaveGPUToHost(const T *d_data, int n) {
  std::vector<T> v(n);
  CUDA_CHECK(cudaMemcpy(v.data(), d_data, sizeof(T) * n, cudaMemcpyDeviceToHost));
  return v;
}

template <typename T>
void SaveGPUToFile(const std::string &file_name, const T *d_data, int n) {
  auto vec = SaveGPUToHost(d_data, n);

  std::ofstream fp(file_name, std::ios::binary);

  if (!fp.is_open()) {
    std::fprintf(stderr, "cannot open file: %s\n", file_name.c_str());
    std::abort();
  }

  fp.write(reinterpret_cast<char *>(vec.data()), sizeof(T) * n);
}

}  // namespace util
