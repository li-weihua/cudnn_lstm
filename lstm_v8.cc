#include "common.h"
#include "util.h"

#include <cuda_fp16.h>
#include <chrono>
#include <iostream>  // debug
#include <string>

template <typename T>
struct DataType {};

template <>
struct DataType<float> {
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  cudnnDataType_t math_precison = CUDNN_DATA_FLOAT;
  cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;
  cudnnRNNAlgo_t rnn_algo = CUDNN_RNN_ALGO_STANDARD;
  std::string suffix = ".0";
};

template <>
struct DataType<half> {
  cudnnDataType_t data_type = CUDNN_DATA_HALF;
  cudnnDataType_t math_precison = CUDNN_DATA_HALF;
  cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
  cudnnRNNAlgo_t rnn_algo = CUDNN_RNN_ALGO_STANDARD;
  std::string suffix = ".1";
};

template <typename T>
struct LSTMWrapper : public DataType<T> {
  // common data
  cudnnHandle_t cudnn_handle;

  int seq_len = 188;
  int batch_size = 8;
  int input_size = 128;
  int hidden_size = 256;
  const int num_layers = 1;

  // It is legal to set projSize equal to hiddenSize, however, in this case,
  // the recurrent projection feature is disabled.
  int proj_size = hidden_size;

  cudnnDataType_t data_type = DataType<T>::data_type;
  cudnnDataType_t math_precison = DataType<T>::math_precison;
  cudnnMathType_t math_type = DataType<T>::math_type;
  cudnnRNNAlgo_t rnn_algo = DataType<T>::rnn_algo;
  std::string suffix = DataType<T>::suffix;

  cudnnRNNDataDescriptor_t xdesc;
  cudnnRNNDataDescriptor_t ydesc;

  cudnnTensorDescriptor_t hdesc;
  cudnnTensorDescriptor_t cdesc;

  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;

  // lstm type
  cudnnRNNMode_t cell_mode = CUDNN_LSTM;
  cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_DOUBLE_BIAS;
  cudnnDirectionMode_t direction_mode = CUDNN_BIDIRECTIONAL;
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
  cudnnForwardMode_t fwd_mode = CUDNN_FWD_MODE_INFERENCE;

  cudnnRNNDescriptor_t rnn_desc;
  cudnnDropoutDescriptor_t dropout_desc = nullptr;

  // input and hx, cx, ...
  T *d_input = nullptr;
  T *d_output = nullptr;
  T *d_hx = nullptr;
  T *d_cx = nullptr;

  int *seq_length_array = nullptr;
  int *dev_seq_lenghts = nullptr;

  size_t weight_space_size = 0;
  size_t workspace_size = 0;
  size_t reserve_space_size = 0;

  T *d_weight_space = nullptr;
  void *d_workspace = nullptr;
  void *d_reserve_space = nullptr;

  LSTMWrapper() {
    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnCreateRNNDataDescriptor(&xdesc));
    CUDNN_CHECK(cudnnCreateRNNDataDescriptor(&ydesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hdesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cdesc));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc));
    CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc));
  }

  ~LSTMWrapper() {
    delete[] seq_length_array;
    CUDA_CHECK(cudaFree(dev_seq_lenghts));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_hx));
    CUDA_CHECK(cudaFree(d_cx));

    if (workspace_size > 0) {
      CUDA_CHECK(cudaFree(d_weight_space));
    }
    if (reserve_space_size > 0) {
      CUDA_CHECK(cudaFree(d_reserve_space));
    }

    CUDNN_CHECK(cudnnDestroyRNNDataDescriptor(xdesc));
    CUDNN_CHECK(cudnnDestroyRNNDataDescriptor(ydesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(hdesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cdesc));
    CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnn_desc));
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn_handle));
  }

  void Init() {
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(T) * seq_len * batch_size * input_size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T) * seq_len * batch_size * 2 * hidden_size));
    CUDA_CHECK(cudaMalloc(&d_hx, sizeof(T) * 2 * batch_size * hidden_size));
    CUDA_CHECK(cudaMalloc(&d_cx, sizeof(T) * 2 * batch_size * hidden_size));

    CUDA_CHECK(cudaMemset(d_hx, 0, sizeof(T) * 2 * batch_size * hidden_size));
    CUDA_CHECK(cudaMemset(d_cx, 0, sizeof(T) * 2 * batch_size * hidden_size));

    // read inputs
    util::ReadRawDataToGPU("../data/input" + suffix + ".data", d_input);

    // sequence length
    seq_length_array = new int[batch_size];
    for (int i = 0; i < batch_size; ++i) {
      seq_length_array[i] = seq_len;
    }

    CUDA_CHECK(cudaMalloc(&dev_seq_lenghts, sizeof(int) * batch_size));
    CUDA_CHECK(cudaMemcpy(dev_seq_lenghts, seq_length_array, sizeof(int) * batch_size,
                          cudaMemcpyHostToDevice));

    CUDNN_CHECK(cudnnSetRNNDataDescriptor(xdesc, data_type, layout, seq_len, batch_size, input_size,
                                          seq_length_array, nullptr));
    CUDNN_CHECK(cudnnSetRNNDataDescriptor(ydesc, data_type, layout, seq_len, batch_size,
                                          hidden_size * 2, seq_length_array, nullptr));

    const int ndims = 3;
    int dim[3];
    int stride[3];

    dim[0] = num_layers * 2;  // bidirection
    dim[1] = batch_size;
    dim[2] = hidden_size;

    stride[0] = dim[2] * dim[1];
    stride[1] = dim[2];
    stride[2] = 1;

    CUDNN_CHECK(cudnnSetTensorNdDescriptor(hdesc, data_type, ndims, dim, stride));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(cdesc, data_type, ndims, dim, stride));

    CUDNN_CHECK(cudnnSetDropoutDescriptor(dropout_desc, cudnn_handle, 0, nullptr, 0, 1));

    CUDNN_CHECK(cudnnSetRNNDescriptor_v8(
        rnn_desc, rnn_algo, cell_mode, bias_mode, direction_mode, input_mode, data_type,
        math_precison, math_type, input_size, hidden_size, proj_size, num_layers, dropout_desc, 0));

    // Dynamic persistent RNN plan
    if (rnn_algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
      // Note: This step is expensive. Once completed the plan can be reused so long as the
      // descriptor
      //       minibatch or datatype don't change.
      CUDNN_CHECK(cudnnBuildRNNDynamic(cudnn_handle, rnn_desc, batch_size));
    }

    // Set up weights and bias parameters
    CUDNN_CHECK(cudnnGetRNNWeightSpaceSize(cudnn_handle, rnn_desc, &weight_space_size));

    std::cout << "weight space size: " << weight_space_size / (1024.0 * 1024.0) << std::endl;

    CUDA_CHECK(cudaMalloc((void **)&d_weight_space, weight_space_size));

    // Set up work space and reserved memory
    CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(cudnn_handle, rnn_desc, fwd_mode, xdesc, &workspace_size,
                                          &reserve_space_size));

    std::cout << "workspace size: " << workspace_size / (1024.0 * 1024.0) << std::endl;
    std::cout << "reserve space size: " << reserve_space_size << std::endl;

    if (workspace_size > 0) CUDA_CHECK(cudaMalloc((void **)&d_workspace, workspace_size));
    if (reserve_space_size > 0)
      CUDA_CHECK(cudaMalloc((void **)&d_reserve_space, reserve_space_size));

    InitWeights();
  }

  void InitWeights() {
    // Initialize Weights
    cudnnTensorDescriptor_t w_desc;
    cudnnTensorDescriptor_t b_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));

    // for LSTM, GEMM number is 8
    const int num_linear_layers = 8;

    for (int layer = 0; layer < num_layers * 2; ++layer) {
      for (int linear_op_id = 0; linear_op_id < num_linear_layers; ++linear_op_id) {
        cudnnDataType_t data_type_temp;
        int nbDims = 0;
        int dim[3], stride[3];
        T *linLayerMat = NULL;
        T *linLayerBias = NULL;

        CUDNN_CHECK(cudnnGetRNNWeightParams(cudnn_handle, rnn_desc, layer, weight_space_size,
                                            d_weight_space, linear_op_id, w_desc,
                                            (void **)&linLayerMat, b_desc, (void **)&linLayerBias));

        if (linLayerMat) {
          CUDNN_CHECK(cudnnGetTensorNdDescriptor(w_desc, 3, &data_type_temp, &nbDims, dim, stride));

          std::string name =
              std::to_string(layer) + std::to_string(linear_op_id) + std::to_string(0);

          util::ReadRawDataToGPU("../data/" + name + suffix + ".data", linLayerMat);
        }

        if (linLayerBias) {
          CUDNN_CHECK(cudnnGetTensorNdDescriptor(b_desc, 3, &data_type_temp, &nbDims, dim, stride));

          std::string name =
              std::to_string(layer) + std::to_string(linear_op_id) + std::to_string(1);

          util::ReadRawDataToGPU("../data/" + name + suffix + ".data", linLayerBias);
        }
      }
    }

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(w_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(b_desc));
  }

  void DoForward() {
    CUDNN_CHECK(cudnnRNNForward(cudnn_handle, rnn_desc, fwd_mode, dev_seq_lenghts, xdesc, d_input,
                                ydesc, d_output, hdesc, d_hx, nullptr, cdesc, d_cx, nullptr,
                                weight_space_size, d_weight_space, workspace_size, d_workspace,
                                reserve_space_size, d_reserve_space));

    // benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float time = 0;

    for (int i = 0; i < 5; ++i) {
      CUDA_CHECK(cudaEventRecord(start));
      CUDNN_CHECK(cudnnRNNForward(cudnn_handle, rnn_desc, fwd_mode, dev_seq_lenghts, xdesc, d_input,
                                  ydesc, d_output, hdesc, d_hx, nullptr, cdesc, d_cx, nullptr,
                                  weight_space_size, d_weight_space, workspace_size, d_workspace,
                                  reserve_space_size, d_reserve_space));

      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

      std::cout << time << " ms" << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  void SaveOutput() {
    util::SaveGPUToFile("../data/output_cudnn" + suffix + ".data", d_output,
                        seq_len * batch_size * 2 * hidden_size);
  }
};

template <typename T>
void Test() {
  LSTMWrapper<T> cudnn_lstm;
  cudnn_lstm.Init();
  cudnn_lstm.DoForward();
  cudnn_lstm.SaveOutput();
}

int main(int argc, char *argv[]) {
  std::cout << "benmarking float-32 preicision" << std::endl;
  Test<float>();
  std::cout << std::endl;

  std::cout << "benmarking half preicision" << std::endl;
  Test<half>();

  return 0;
}
