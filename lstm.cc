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
  cudnnRNNAlgo_t rnn_algo = CUDNN_RNN_ALGO_STANDARD;  // best algorithm
  std::string suffix = ".0";
};

template <>
struct DataType<half> {
  cudnnDataType_t data_type = CUDNN_DATA_HALF;
  cudnnDataType_t math_precison = CUDNN_DATA_HALF;
  cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
  cudnnRNNAlgo_t rnn_algo = CUDNN_RNN_ALGO_STANDARD;  // best algorithm
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

  cudnnTensorDescriptor_t *xdesc;
  cudnnTensorDescriptor_t *ydesc;
  cudnnTensorDescriptor_t hdesc;
  cudnnTensorDescriptor_t cdesc;
  cudnnFilterDescriptor_t wdesc;

  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;

  // lstm type
  cudnnRNNMode_t cell_mode = CUDNN_LSTM;
  cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_DOUBLE_BIAS;
  cudnnDirectionMode_t direction_mode = CUDNN_BIDIRECTIONAL;
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;

  cudnnRNNDescriptor_t rnn_desc;
  cudnnDropoutDescriptor_t dropout_desc = nullptr;

  // input and hx, cx, ...
  T *d_input = nullptr;
  T *d_output = nullptr;
  T *d_hx = nullptr;
  T *d_cx = nullptr;
  T *d_weights = nullptr;

  int *seq_length_array = nullptr;

  size_t weight_space_size = 0;
  size_t workspace_size = 0;
  size_t reserve_space_size = 0;

  T *d_weight_space = nullptr;
  void *d_workspace = nullptr;
  void *d_reserve_space = nullptr;

  LSTMWrapper() {
    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc));
    CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hdesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cdesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc));
  }

  ~LSTMWrapper() {
    delete[] seq_length_array;
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

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(hdesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cdesc));
    CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnn_desc));
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn_handle));
  }

  void Init() {
    xdesc = (cudnnTensorDescriptor_t *)malloc(seq_len * sizeof(cudnnTensorDescriptor_t));
    ydesc = (cudnnTensorDescriptor_t *)malloc(seq_len * sizeof(cudnnTensorDescriptor_t));

    const int ndims = 3;
    int dim_x[3];
    int dim_y[3];
    int stride_x[3];
    int stride_y[3];

    dim_x[0] = batch_size;
    dim_x[1] = input_size;
    dim_x[2] = 1;

    stride_x[0] = dim_x[1] * dim_x[2];
    stride_x[1] = dim_x[2];
    stride_x[2] = 1;

    dim_y[0] = batch_size;
    dim_y[1] = hidden_size * 2;
    dim_y[2] = 1;

    stride_y[0] = dim_y[1] * dim_y[2];
    stride_y[1] = dim_y[2];
    stride_y[2] = 1;

    for (int i = 0; i < seq_len; i++) {
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&xdesc[i]));
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&ydesc[i]));

      CUDNN_CHECK(cudnnSetTensorNdDescriptor(xdesc[i], data_type, 3, dim_x, stride_x));
      CUDNN_CHECK(cudnnSetTensorNdDescriptor(ydesc[i], data_type, 3, dim_y, stride_y));
    }

    int dim_h[3];
    int stride_h[3];

    dim_h[0] = num_layers * 2;  // bidirection
    dim_h[1] = batch_size;
    dim_h[2] = hidden_size;

    stride_h[0] = dim_h[2] * dim_h[1];
    stride_h[1] = dim_h[2];
    stride_h[2] = 1;

    CUDNN_CHECK(cudnnSetTensorNdDescriptor(hdesc, data_type, ndims, dim_h, stride_h));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(cdesc, data_type, ndims, dim_h, stride_h));

    // set dropout
    CUDNN_CHECK(cudnnSetDropoutDescriptor(dropout_desc, cudnn_handle, 0, nullptr, 0, 1));

    CUDA_CHECK(cudaMalloc(&d_input, sizeof(T) * seq_len * batch_size * input_size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T) * seq_len * batch_size * 2 * hidden_size));
    CUDA_CHECK(cudaMalloc(&d_hx, sizeof(T) * 2 * batch_size * hidden_size));
    CUDA_CHECK(cudaMalloc(&d_cx, sizeof(T) * 2 * batch_size * hidden_size));

    CUDA_CHECK(cudaMemset(d_hx, 0, sizeof(T) * 2 * batch_size * hidden_size));
    CUDA_CHECK(cudaMemset(d_cx, 0, sizeof(T) * 2 * batch_size * hidden_size));

    // read inputs
    util::ReadRawDataToGPU("../data/input" + suffix + ".data", d_input);

    CUDNN_CHECK(cudnnSetRNNDescriptor_v6(cudnn_handle, rnn_desc, hidden_size, num_layers,
                                         dropout_desc, input_mode, direction_mode, cell_mode,
                                         rnn_algo, data_type));

    cudnnPersistentRNNPlan_t rnn_plan;
    if (rnn_algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
      // Note: This step is expensive. Once completed the plan can be reused so long as the
      // descriptor
      //       minibatch or datatype don't change.
      CUDNN_CHECK(cudnnCreatePersistentRNNPlan(rnn_desc, batch_size, data_type, &rnn_plan));
      // Tell calls using this descriptor which plan to use.
      CUDNN_CHECK(cudnnSetPersistentRNNPlan(rnn_desc, rnn_plan));
    }

    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(cudnn_handle, rnn_desc, seq_len, xdesc, &workspace_size));

    std::cout << "workspace size: " << workspace_size / (1024.0 * 1024.0) << std::endl;

    if (workspace_size > 0) CUDA_CHECK(cudaMalloc((void **)&d_workspace, workspace_size));

    InitWeights();
  }

  void InitWeights() {
    // Initialize Weights

    size_t weights_size;
    CUDNN_CHECK(cudnnGetRNNParamsSize(cudnn_handle, rnn_desc, xdesc[0], &weights_size, data_type));

    int dim[3];
    dim[0] = weights_size / sizeof(T);
    dim[1] = 1;
    dim[2] = 1;

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(wdesc, data_type, CUDNN_TENSOR_NCHW, 3, dim));
    CUDA_CHECK(cudaMalloc((void **)&d_weights, weights_size));

    // for LSTM, GEMM number is 8
    const int num_linear_layers = 8;

    for (int layer = 0; layer < num_layers * 2; ++layer) {
      for (int lin_layer_id = 0; lin_layer_id < num_linear_layers; ++lin_layer_id) {
        cudnnDataType_t dataType;
        cudnnTensorFormat_t format;
        int nbDims;
        int filterDimA[3];

        // Initialize layer weights
        cudnnFilterDescriptor_t linLayerMatDesc;
        T *linLayerMat;

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&linLayerMatDesc));
        CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(cudnn_handle, rnn_desc, layer, xdesc[0], wdesc,
                                                    d_weights, lin_layer_id, linLayerMatDesc,
                                                    (void **)&linLayerMat));

        CUDNN_CHECK(cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, &dataType, &format, &nbDims,
                                               filterDimA));
        std::string weight_name =
            std::to_string(layer) + std::to_string(lin_layer_id) + std::to_string(0);

        util::ReadRawDataToGPU("../data/" + weight_name + suffix + ".data", linLayerMat);

        CUDNN_CHECK(cudnnDestroyFilterDescriptor(linLayerMatDesc));

        // Initialize layer bias
        cudnnFilterDescriptor_t linLayerBiasDesc;
        T *linLayerBias;

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
        CUDNN_CHECK(cudnnGetRNNLinLayerBiasParams(cudnn_handle, rnn_desc, layer, xdesc[0], wdesc,
                                                  d_weights, lin_layer_id, linLayerBiasDesc,
                                                  (void **)&linLayerBias));

        CUDNN_CHECK(cudnnGetFilterNdDescriptor(linLayerBiasDesc, 3, &dataType, &format, &nbDims,
                                               filterDimA));
        std::string bias_name =
            std::to_string(layer) + std::to_string(lin_layer_id) + std::to_string(1);

        util::ReadRawDataToGPU("../data/" + bias_name + suffix + ".data", linLayerBias);

        CUDNN_CHECK(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
      }
    }
  }

  // warm up
  void WarmUp() {
    for (int i = 0; i < 3; ++i)
      CUDNN_CHECK(cudnnRNNForwardInference(
          /*handle=*/cudnn_handle, /*rnnDesc=*/rnn_desc, /*seqLength=*/seq_len, /*xDesc=*/xdesc,
          /*x=*/d_input,
          /*hxDesc=*/hdesc, /*hx=*/d_hx, /*cxDesc=*/cdesc, /*cx=*/d_cx, /*wDesc=*/wdesc,
          /*w=*/d_weights, /*yDesc=*/ydesc, /*y=*/d_output,
          /*hyDesc=*/hdesc, /*hy=*/nullptr, /*cyDesc=*/cdesc,
          /*cy=*/nullptr, /*workspace=*/d_workspace, /*workSpaceSizeInBytes=*/workspace_size));
  }

  // benchmark
  void DoForward() {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float time = 0;

    for (int i = 0; i < 5; ++i) {
      CUDA_CHECK(cudaEventRecord(start));

      CUDNN_CHECK(cudnnRNNForwardInference(
          /*handle=*/cudnn_handle, /*rnnDesc=*/rnn_desc, /*seqLength=*/seq_len, /*xDesc=*/xdesc,
          /*x=*/d_input,
          /*hxDesc=*/hdesc, /*hx=*/d_hx, /*cxDesc=*/cdesc, /*cx=*/d_cx, /*wDesc=*/wdesc,
          /*w=*/d_weights, /*yDesc=*/ydesc, /*y=*/d_output,
          /*hyDesc=*/hdesc, /*hy=*/nullptr, /*cyDesc=*/cdesc,
          /*cy=*/nullptr, /*workspace=*/d_workspace, /*workSpaceSizeInBytes=*/workspace_size));

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
  cudnn_lstm.WarmUp();
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
