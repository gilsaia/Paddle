/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/inference/tensorrt/plugin/matmul_op_int4_plugin.h"
// #include "cublas_v2.h"
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

char const* MATMULINT4PLUGINVERSION{"1"};
char const* MATMULINT4PLUGINNAME{"MatmulInt4Plugin"};

// MatmulInt4Plugin::MatmulInt4Plugin(nvinfer1::Dims const& dims_x,
//                                    nvinfer1::DataType type_x,
//                                    nvinfer1::Dims const& dims_y,
//                                    nvinfer1::DataType type_y,
//                                    Int4GemmActivationType activation_type,
//                                    void* y)
//     : dims_x_(dims_x),
//       dims_y_(dims_y),
//       type_x_(type_x),
//       type_y_(type_y),
//       activation_type_(activation_type),
//       y_ori_(y) {
//   m_ = dims_x.d[dims_x.nbDims - 2];
//   n_ = dims_y.d[dims_y.nbDims - 2];
//   k_ = dims_y.d[dims_y.nbDims - 1];
//   uint64_t mk = m_ * k_;
//   uint64_t kn = k_ * n_;
//   uint64_t mn = m_ * n_;
//   batch_ = 1;
//   for (int i = 0; i < dims_x.nbDims - 2; ++i) {
//     batch_ *= dims_x.d[i];
//   }
//   uint64_t y_size = kn * sizeof(type_y);
//   cudaMalloc(&y_device_, y_size);
//   cudaMemcpy(y_device_, y, y_size, cudaMemcpyHostToDevice);

//   cudaMalloc(reinterpret_cast<void**>(&x_convert_), mk / 2);
//   cudaMalloc(reinterpret_cast<void**>(&y_convert_), kn / 2);
//   cudaMalloc(reinterpret_cast<void**>(&y_extra_), kn * 4);
//   cudaMalloc(reinterpret_cast<void**>(&res_), mn * 4);
//   if (type_y_ == nvinfer1::DataType::kHALF) {
//     phi::fusion::cutlass_gemm_internal::ConvertData<cutlass::half_t,
//     int32_t>(
//         static_cast<cutlass::half_t*>(y_device_), y_extra_, k_ * n_);
//   } else if (type_y_ == nvinfer1::DataType::kINT8) {
//     phi::fusion::cutlass_gemm_internal::ConvertData<int8_t, int32_t>(
//         static_cast<int8_t*>(y_device_), y_extra_, k_ * n_);
//   } else if (type_y_ == nvinfer1::DataType::kINT32) {
//     phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int32_t>(
//         static_cast<int32_t*>(y_device_), y_extra_, k_ * n_);
//   } else if (type_y_ == nvinfer1::DataType::kFLOAT) {
//     phi::fusion::cutlass_gemm_internal::ConvertData<float, int32_t>(
//         static_cast<float*>(y_device_), y_extra_, k_ * n_);
//   }
//   phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<int32_t>(
//       y_extra_, y_convert_, k_ * n_);
// }

MatmulInt4Plugin::MatmulInt4Plugin(nvinfer1::Dims const& dims_x,
                                   nvinfer1::DataType type_x,
                                   float scale_x,
                                   nvinfer1::Dims const& dims_y,
                                   nvinfer1::DataType type_y,
                                   float scale_y,
                                   Int4GemmActivationType activation_type,
                                   bool output_int4_range,
                                   bool with_bias,
                                   nvinfer1::DataType type_bias,
                                   float scale_bias,
                                   float scale_out,
                                   void* y,
                                   void* bias)
    : dims_x_(dims_x),
      dims_y_(dims_y),
      scale_x_(scale_x),
      type_x_(type_x),
      type_y_(type_y),
      scale_y_(scale_y),
      activation_type_(activation_type),
      output_int4_range_(output_int4_range),
      with_bias_(with_bias),
      type_bias_(type_bias),
      scale_bias_(scale_bias),
      scale_out_(scale_out),
      y_device_(y),
      bias_device_(bias),
      max_m_(-1) {
  m_ = dims_x.d[dims_x.nbDims - 2];
  n_ = dims_y.d[dims_y.nbDims - 1];
  k_ = dims_y.d[dims_y.nbDims - 2];
  batch_ = 1;
  // uint64_t mk = m_ * k_;
  // uint64_t kn = k_ * n_;
  // uint64_t mn = m_ * n_;
  // for (int i = 0; i < dims_x.nbDims - 2; ++i) {
  //   batch_ *= dims_x.d[i];
  // }

  // cudaMalloc(reinterpret_cast<void**>(&x_convert_), mk / 2);
  // cudaMalloc(reinterpret_cast<void**>(&y_convert_), kn / 2);
  // cudaMalloc(reinterpret_cast<void**>(&x_extra_), mk * 4);
  // cudaMalloc(reinterpret_cast<void**>(&y_extra_), kn * 4);
  // cudaMalloc(reinterpret_cast<void**>(&res_), mn * 4);
  // if (type_y_ == nvinfer1::DataType::kHALF) {
  //   phi::fusion::cutlass_gemm_internal::ConvertData<cutlass::half_t,
  //   int32_t>(
  //       static_cast<cutlass::half_t*>(y_device_), y_extra_, k_ * n_);
  // } else if (type_y_ == nvinfer1::DataType::kINT8) {
  //   phi::fusion::cutlass_gemm_internal::ConvertData<int8_t, int32_t>(
  //       static_cast<int8_t*>(y_device_), y_extra_, k_ * n_);
  // } else if (type_y_ == nvinfer1::DataType::kINT32) {
  //   phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int32_t>(
  //       static_cast<int32_t*>(y_device_), y_extra_, k_ * n_);
  // } else if (type_y_ == nvinfer1::DataType::kFLOAT) {
  //   phi::fusion::cutlass_gemm_internal::ConvertData<float, int32_t>(
  //       static_cast<float*>(y_device_), y_extra_, k_ * n_);
  // }
  // phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<int32_t>(
  //     y_extra_, y_convert_, k_ * n_);

  // if (with_bias_) {
  //   cudaMalloc(reinterpret_cast<void**>(&bias_convert_), mn * 4);
  //   if (type_bias_ == nvinfer1::DataType::kHALF) {
  //     phi::fusion::cutlass_gemm_internal::ConvertData<cutlass::half_t,
  //     int32_t>(
  //         static_cast<cutlass::half_t*>(bias_device_), bias_convert_, m_ *
  //         n_);
  //   } else if (type_bias_ == nvinfer1::DataType::kFLOAT) {
  //     phi::fusion::cutlass_gemm_internal::ConvertData<float, int32_t>(
  //         static_cast<float*>(bias_device_), bias_convert_, m_ * n_);
  //   } else if (type_bias_ == nvinfer1::DataType::kINT8) {
  //     phi::fusion::cutlass_gemm_internal::ConvertData<int8_t, int32_t>(
  //         static_cast<int8_t*>(bias_device_), bias_convert_, m_ * n_);
  //   } else if (type_bias_ == nvinfer1::DataType::kINT32) {
  //     phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int32_t>(
  //         static_cast<int32_t*>(bias_device_), bias_convert_, m_ * n_);
  //   }
  // }

  // int32_t* debug = reinterpret_cast<int32_t*>(malloc(k_ * n_ * 4));
  // std::cout << "in construct weight print" << std::endl;
  // cudaMemcpy(debug, y_extra_, k_ * n_ * 4, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 20; ++i) {
  //   for (int j = 0; j < n_; ++j) {
  //     std::cout << debug[i * n_ + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // free(debug);
}

MatmulInt4Plugin::MatmulInt4Plugin(void const* data, size_t length) {
  DeserializeValue(&data, &length, &dims_x_);
  DeserializeValue(&data, &length, &dims_y_);
  DeserializeValue(&data, &length, &type_x_);
  DeserializeValue(&data, &length, &type_y_);
  DeserializeValue(&data, &length, &scale_x_);
  DeserializeValue(&data, &length, &scale_y_);
  DeserializeValue(&data, &length, &activation_type_);
  DeserializeValue(&data, &length, &output_int4_range_);
  DeserializeValue(&data, &length, &with_bias_);
  DeserializeValue(&data, &length, &type_bias_);
  DeserializeValue(&data, &length, &scale_bias_);
  DeserializeValue(&data, &length, &scale_out_);
  DeserializeValue(&data, &length, &batch_);
  DeserializeValue(&data, &length, &m_);
  DeserializeValue(&data, &length, &n_);
  DeserializeValue(&data, &length, &k_);
  char const* d = static_cast<char const*>(data);
  cudaMalloc(&y_device_, n_ * k_ * sizeof(type_y_));
  cudaMemcpy(y_device_, d, k_ * n_ * sizeof(type_y_), cudaMemcpyHostToDevice);
  max_m_ = -1;
  if (with_bias_) {
    char const* d = static_cast<char const*>(data) + k_ * n_ * sizeof(type_y_);
    uint64_t bias_size = n_ * sizeof(type_bias_);
    cudaMalloc(&bias_device_, bias_size);
    cudaMemcpy(bias_device_, d, bias_size, cudaMemcpyHostToDevice);
  }
  // uint64_t mk = m_ * k_;
  // uint64_t kn = k_ * n_;
  // uint64_t mn = m_ * n_;
  // cudaMalloc(reinterpret_cast<void**>(&x_convert_), mk / 2);
  // cudaMalloc(reinterpret_cast<void**>(&y_convert_), kn / 2);
  // cudaMalloc(reinterpret_cast<void**>(&x_extra_), mk * 4);
  // cudaMalloc(reinterpret_cast<void**>(&y_extra_), kn * 4);
  // cudaMalloc(reinterpret_cast<void**>(&res_), mn * 4);
  // if (type_y_ == nvinfer1::DataType::kHALF) {
  //   phi::fusion::cutlass_gemm_internal::ConvertData<cutlass::half_t,
  //   int32_t>(
  //       static_cast<cutlass::half_t*>(y_device_), y_extra_, k_ * n_);
  // } else if (type_y_ == nvinfer1::DataType::kINT8) {
  //   phi::fusion::cutlass_gemm_internal::ConvertData<int8_t, int32_t>(
  //       static_cast<int8_t*>(y_device_), y_extra_, k_ * n_);
  // } else if (type_y_ == nvinfer1::DataType::kINT32) {
  //   phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int32_t>(
  //       static_cast<int32_t*>(y_device_), y_extra_, k_ * n_);
  // } else if (type_y_ == nvinfer1::DataType::kFLOAT) {
  //   phi::fusion::cutlass_gemm_internal::ConvertData<float, int32_t>(
  //       static_cast<float*>(y_device_), y_extra_, k_ * n_);
  // }
  // phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<int32_t>(
  //     y_extra_, y_convert_, k_ * n_);
  // if (with_bias_) {
  //   cudaMalloc(reinterpret_cast<void**>(&bias_convert_), mn * 4);
  //   if (type_bias_ == nvinfer1::DataType::kHALF) {
  //     phi::fusion::cutlass_gemm_internal::ConvertData<cutlass::half_t,
  //     int32_t>(
  //         static_cast<cutlass::half_t*>(bias_device_), bias_convert_, m_ *
  //         n_);
  //   } else if (type_bias_ == nvinfer1::DataType::kFLOAT) {
  //     phi::fusion::cutlass_gemm_internal::ConvertData<float, int32_t>(
  //         static_cast<float*>(bias_device_), bias_convert_, m_ * n_);
  //   } else if (type_bias_ == nvinfer1::DataType::kINT8) {
  //     phi::fusion::cutlass_gemm_internal::ConvertData<int8_t, int32_t>(
  //         static_cast<int8_t*>(bias_device_), bias_convert_, m_ * n_);
  //   } else if (type_bias_ == nvinfer1::DataType::kINT32) {
  //     phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int32_t>(
  //         static_cast<int32_t*>(bias_device_), bias_convert_, m_ * n_);
  //   }
  // }
}

void MatmulInt4Plugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nb_inputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nb_outputs) noexcept {
  // std::cerr << "in configure plugin" << std::endl;
  auto& x_dims = in->desc.dims;
  bool change = false;
  if (x_dims.nbDims == 4 && x_dims.d[2] == 1 && x_dims.d[3] == 1) {
    int now_max = in->max.d[0];
    if (now_max > max_m_) {
      if (max_m_ != -1) {
        // change max dimesion,free space first
        cudaFree(x_convert_);
        cudaFree(y_convert_);
        cudaFree(y_extra_);
        cudaFree(res_);
        if (with_bias_) {
          cudaFree(bias_extra_);
          cudaFree(bias_convert_);
        }
      }
      max_m_ = now_max;
      m_ = now_max;
      change = true;
    }
  } else if (x_dims.nbDims == 3) {
    int now_max = in->max.d[1];
    if (now_max > max_m_) {
      if (max_m_ != -1) {
        cudaFree(x_convert_);
        cudaFree(y_convert_);
        cudaFree(y_extra_);
        cudaFree(res_);
        if (with_bias_) {
          cudaFree(bias_extra_);
          cudaFree(bias_convert_);
        }
      }
      max_m_ = now_max;
      m_ = now_max;
      change = true;
    }
  }
  if (change) {
    uint64_t mk = m_ * k_;
    uint64_t kn = k_ * n_;
    uint64_t mn = m_ * n_;
    cudaMalloc(reinterpret_cast<void**>(&x_convert_), mk / 2);
    cudaMalloc(reinterpret_cast<void**>(&y_convert_), kn / 2);
    cudaMalloc(reinterpret_cast<void**>(&y_extra_), kn * 4);
    cudaMalloc(reinterpret_cast<void**>(&res_), mn * 4);
    if (type_y_ == nvinfer1::DataType::kHALF) {
      phi::fusion::cutlass_gemm_internal::ConvertData<cutlass::half_t, int32_t>(
          static_cast<cutlass::half_t*>(y_device_), y_extra_, k_ * n_);
    } else if (type_y_ == nvinfer1::DataType::kINT8) {
      phi::fusion::cutlass_gemm_internal::ConvertData<int8_t, int32_t>(
          static_cast<int8_t*>(y_device_), y_extra_, k_ * n_);
    } else if (type_y_ == nvinfer1::DataType::kINT32) {
      phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int32_t>(
          static_cast<int32_t*>(y_device_), y_extra_, k_ * n_);
    } else if (type_y_ == nvinfer1::DataType::kFLOAT) {
      phi::fusion::cutlass_gemm_internal::ConvertData<float, int32_t>(
          static_cast<float*>(y_device_), y_extra_, k_ * n_);
    }
    phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<int32_t>(
        y_extra_, y_convert_, k_ * n_);
    if (with_bias_) {
      cudaMalloc(reinterpret_cast<void**>(&bias_extra_), n_ * 4);
      cudaMalloc(reinterpret_cast<void**>(&bias_convert_), mn * 4);
      if (type_bias_ == nvinfer1::DataType::kHALF) {
        phi::fusion::cutlass_gemm_internal::ConvertData<cutlass::half_t,
                                                        int32_t>(
            static_cast<cutlass::half_t*>(bias_device_), bias_extra_, n_);
      } else if (type_bias_ == nvinfer1::DataType::kFLOAT) {
        phi::fusion::cutlass_gemm_internal::ConvertData<float, int32_t>(
            static_cast<float*>(bias_device_), bias_extra_, n_);
      } else if (type_bias_ == nvinfer1::DataType::kINT8) {
        phi::fusion::cutlass_gemm_internal::ConvertData<int8_t, int32_t>(
            static_cast<int8_t*>(bias_device_), bias_extra_, n_);
      } else if (type_bias_ == nvinfer1::DataType::kINT32) {
        phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int32_t>(
            static_cast<int32_t*>(bias_device_), bias_extra_, n_);
      }
      dim3 gridb(128);
      dim3 blockb(768);
      phi::fusion::cutlass_gemm_internal::ExpendKernel<int32_t>
          <<<gridb, blockb>>>(reinterpret_cast<const int32_t*>(bias_extra_),
                              reinterpret_cast<int32_t*>(bias_convert_),
                              n_,
                              m_,
                              0);
    }
  }
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "after configure wrong"
              << " " << cudaGetErrorString(error) << std::endl;
  }
}

bool MatmulInt4Plugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) noexcept {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    1,
                    platform::errors::InvalidArgument("Must have 2 inputs, "
                                                      "but got %d input(s). ",
                                                      nb_inputs));
  PADDLE_ENFORCE_EQ(nb_outputs,
                    1,
                    platform::errors::InvalidArgument("Must have 1 output, "
                                                      "but got %d output(s). ",
                                                      nb_outputs));
  if (pos == 0) {
    // return (in_out[pos].type == nvinfer1::DataType::kHALF ||
    //         in_out[pos].type == nvinfer1::DataType::kINT8 ||
    //         in_out[pos].type == nvinfer1::DataType::kFLOAT ||
    //         in_out[pos].type == nvinfer1::DataType::kINT32) &&
    //        in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
    return in_out[pos].type == nvinfer1::DataType::kINT8 &&
           in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
  } else if (pos == 1) {
    return in_out[pos].type == nvinfer1::DataType::kINT8 &&
           in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
  return false;
}

nvinfer1::DataType MatmulInt4Plugin::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* input_types,
    int32_t nb_inputs) const noexcept {
  return nvinfer1::DataType::kINT8;
}

void MatmulInt4Plugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) noexcept {}

void MatmulInt4Plugin::detachFromContext() noexcept {}

nvinfer1::IPluginV2DynamicExt* MatmulInt4Plugin::clone() const noexcept {
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "before clone wrong"
              << " " << cudaGetErrorString(error) << std::endl;
  }
  auto p = new MatmulInt4Plugin(dims_x_,
                                type_x_,
                                scale_x_,
                                dims_y_,
                                type_y_,
                                scale_y_,
                                activation_type_,
                                output_int4_range_,
                                with_bias_,
                                type_bias_,
                                scale_bias_,
                                scale_out_,
                                y_device_,
                                bias_device_);
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "after clone wrong"
              << " " << cudaGetErrorString(error) << std::endl;
  }
  p->setPluginNamespace(namespace_.c_str());
  return p;
}

const char* MatmulInt4Plugin::getPluginType() const noexcept {
  return MATMULINT4PLUGINNAME;
}

const char* MatmulInt4Plugin::getPluginVersion() const noexcept {
  return MATMULINT4PLUGINVERSION;
}

int32_t MatmulInt4Plugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs MatmulInt4Plugin::getOutputDimensions(
    int32_t output_index,
    nvinfer1::DimsExprs const* inputs,
    int32_t nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
  nvinfer1::DimsExprs output_dims;
  output_dims.nbDims = inputs[0].nbDims;
  for (int i = 0; i < inputs[0].nbDims; ++i) {
    output_dims.d[i] = inputs[0].d[i];
  }
  if (inputs[0].nbDims == 4) {
    output_dims.d[1] = expr_builder.constant(n_);
  } else if (inputs[0].nbDims == 3) {
    output_dims.d[2] = expr_builder.constant(n_);
  }
  // output_dims.d[output_dims.nbDims - 2] = m_;
  // output_dims.d[output_dims.nbDims - 1] = expr_builder.constant(n_);
  return output_dims;
}

int32_t MatmulInt4Plugin::initialize() noexcept { return 0; }

void MatmulInt4Plugin::terminate() noexcept {
  if (max_m_ == -1) {
    return;
  }
  // cudaDeviceSynchronize();
  // cudaError_t error = cudaGetLastError();
  // if (error != cudaSuccess) {
  //   std::cout << "before terminate wrong"
  //             << " " << cudaGetErrorString(error) << std::endl;
  // }
  cudaFree(reinterpret_cast<void*>(x_convert_));
  // cudaDeviceSynchronize();
  // error = cudaGetLastError();
  // if (error != cudaSuccess) {
  //   std::cout << "before Ac wrong"
  //             << " " << cudaGetErrorString(error) << std::endl;
  // }
  cudaFree(reinterpret_cast<void*>(y_convert_));
  cudaFree(reinterpret_cast<void*>(y_extra_));
  // cudaDeviceSynchronize();
  // error = cudaGetLastError();
  // if (error != cudaSuccess) {
  //   std::cout << "before Bc wrong"
  //             << " " << cudaGetErrorString(error) << std::endl;
  // }
  cudaFree(reinterpret_cast<void*>(res_));
  // cudaDeviceSynchronize();
  // error = cudaGetLastError();
  // if (error != cudaSuccess) {
  //   std::cout << "before Cr wrong"
  //             << " " << cudaGetErrorString(error) << std::endl;
  // }
  // cudaFree(y_device_);
  if (with_bias_) {
    // cudaFree(bias_device_);
    cudaFree(bias_extra_);
    cudaFree(reinterpret_cast<void*>(bias_convert_));
  }

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "after terminate wrong"
              << " " << cudaGetErrorString(error) << std::endl;
  }
}

size_t MatmulInt4Plugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* input_desc,
    int32_t nb_inputs,
    nvinfer1::PluginTensorDesc const* output_desc,
    int32_t nb_outputs) const noexcept {
  return 0;
}

size_t MatmulInt4Plugin::getSerializationSize() const noexcept {
  return SerializedSize(dims_x_) + SerializedSize(dims_y_) +
         SerializedSize(type_x_) + SerializedSize(type_y_) +
         SerializedSize(type_bias_) + SerializedSize(activation_type_) +
         SerializedSize(output_int4_range_) + SerializedSize(with_bias_) +
         SerializedSize(batch_) + SerializedSize(m_) + SerializedSize(n_) +
         SerializedSize(k_) + SerializedSize(scale_x_) +
         SerializedSize(scale_y_) + SerializedSize(scale_bias_) +
         SerializedSize(scale_out_) + n_ * k_ * sizeof(type_y_) +
         (with_bias_ ? n_ * sizeof(type_bias_) : 0);
}

void MatmulInt4Plugin::serialize(void* buffer) const noexcept {
  std::cout << "in plugin serialize" << std::endl;
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "before serialize wrong"
              << " " << cudaGetErrorString(error) << std::endl;
  }
  SerializeValue(&buffer, dims_x_);
  SerializeValue(&buffer, dims_y_);
  SerializeValue(&buffer, type_x_);
  SerializeValue(&buffer, type_y_);
  SerializeValue(&buffer, scale_x_);
  SerializeValue(&buffer, scale_y_);
  SerializeValue(&buffer, activation_type_);
  SerializeValue(&buffer, output_int4_range_);
  SerializeValue(&buffer, with_bias_);
  SerializeValue(&buffer, type_bias_);
  SerializeValue(&buffer, scale_bias_);
  SerializeValue(&buffer, scale_out_);
  SerializeValue(&buffer, batch_);
  SerializeValue(&buffer, m_);
  SerializeValue(&buffer, n_);
  SerializeValue(&buffer, k_);
  void* d = static_cast<char*>(buffer);
  SerializeCudaPointer<char>(
      &d, static_cast<char*>(y_device_), n_ * k_ * sizeof(type_y_));
  if (with_bias_) {
    SerializeCudaPointer<char>(
        &d, static_cast<char*>(bias_device_), n_ * sizeof(type_bias_));
  }
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "after serialize wrong"
              << " " << cudaGetErrorString(error) << std::endl;
  }
}

void MatmulInt4Plugin::destroy() noexcept { delete this; }

char const* MatmulInt4Plugin::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

void MatmulInt4Plugin::setPluginNamespace(
    char const* plugin_name_space) noexcept {
  namespace_ = plugin_name_space;
}

int32_t MatmulInt4Plugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                  nvinfer1::PluginTensorDesc const* output_desc,
                                  void const* const* inputs,
                                  void* const* outputs,
                                  void* workspace,
                                  cudaStream_t stream) noexcept {
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  platform::CUDAPlace place(platform::GetCurrentDeviceId());

  auto* device_context = static_cast<phi::GPUContext*>(pool.Get(place));
  const phi::GPUContext& dev_ctx = *device_context;

  auto& dims_x = input_desc[0].dims;
  if (dims_x.nbDims == 4 && dims_x.d[2] == 1 && dims_x.d[3] == 1) {
    m_ = dims_x.d[0];
  } else if (dims_x.nbDims == 3) {
    m_ = dims_x.d[1];
  }

  float scale_alpha = 1;
  if (output_int4_range_) {
    // scale_alpha=(scale_x_/127)*(scale_y_/7)/(scale_out_/7);
    scale_alpha = scale_x_ * scale_y_ / scale_out_ / 127;
  } else {
    // float scale_alpha = (scale_x_/127) * (scale_y_/7) / (scale_out_/127);
    scale_alpha = scale_x_ * scale_y_ / scale_out_ / 7;
  }
  // scale_alpha = 1;

  float scale_beta = 0;
  if (output_int4_range_) {
    // float scale_beta = (scale_bias_ / 127) / (scale_out_ / 7);
    scale_beta = scale_bias_ * 7 / scale_out_ / 127;
  } else {
    // float scale_beta = (scale_bias_ / 127) / (scale_out_ / 127);
    scale_beta = scale_bias_ / scale_out_;
  }
  // if (n_ == 768) {
  //   scale_beta = 0;
  // }
  // scale_beta = 0;

  // void* y_tmp;
  // cudaMalloc(&y_tmp, k_ * n_);
  // phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int8_t>(
  //     y_extra_, static_cast<int8_t*>(y_tmp), k_ * n_);

  // auto handle = dev_ctx.cublas_handle();
  // auto status = paddle::platform::dynload::cublasGemmEx(handle,
  //                                                       CUBLAS_OP_T,
  //                                                       CUBLAS_OP_N,
  //                                                       m_,
  //                                                       n_,
  //                                                       k_,
  //                                                       &scale_alpha,
  //                                                       inputs[0],
  //                                                       CUDA_R_8I,
  //                                                       k_,
  //                                                       y_tmp,
  //                                                       CUDA_R_8I,
  //                                                       k_,
  //                                                       &scale_beta,
  //                                                       bias_convert_,
  //                                                       CUDA_R_32F,
  //                                                       m_,
  //                                                       CUBLAS_COMPUTE_32F,
  //                                                       CUBLAS_GEMM_DEFAULT);
  // float* tmp = reinterpret_cast<float*>(bias_convert_);
  // phi::fusion::cutlass_gemm_internal::ConvertData<float, int32_t>(
  //     tmp, res_, m_ * n_);

  // cudaFree(y_tmp);

  const int8_t* x = static_cast<const int8_t*>(inputs[0]);
  phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<int8_t>(
      x, x_convert_, m_ * k_, dev_ctx.stream());
  // phi::fusion::cutlass_gemm_internal::ConvertDataToInt4WithScale(
  //     x, x_extra_, x_convert_, m_ * k_, scale_x_);

  // auto* res = static_cast<int32_t*>(outputs[0]);
  phi::fusion::cutlass_gemm_internal::GemmAllParams params{x_convert_,
                                                           y_convert_,
                                                           bias_convert_,
                                                           res_,
                                                           batch_,
                                                           m_,
                                                           n_,
                                                           k_,
                                                           scale_alpha,
                                                           scale_beta,
                                                           &dev_ctx};
  int sm = phi::fusion::cutlass_gemm_internal::getSMVersion();
  if (activation_type_ == INT4_GEMM_ACTIVATION_TYPE_NONE) {
    phi::fusion::cutlass_gemm_internal::Int4Gemm(params, sm);
  } else if (activation_type_ == INT4_GEMM_ACTIVATION_TYPE_RELU) {
    phi::fusion::cutlass_gemm_internal::Int4GemmRelu(params, sm);
  } else if (activation_type_ == INT4_GEMM_ACTIVATION_TYPE_BIAS) {
    phi::fusion::cutlass_gemm_internal::Int4GemmBias(params, sm);
  } else if (activation_type_ == INT4_GEMM_ACTIVATION_TYPE_BIAS_RELU) {
    phi::fusion::cutlass_gemm_internal::Int4GemmBiasRelu(params, sm);
  } else {
    // should not in here
    phi::fusion::cutlass_gemm_internal::Int4Gemm(params, sm);
  }

  auto* res = static_cast<int8_t*>(outputs[0]);
  phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int8_t>(
      res_, res, m_ * n_, dev_ctx.stream());

  // std::cout << "input scale " << scale_x_ << "output scale" << scale_out_
  //           << "weight scale" << scale_y_ << "bias scale" << scale_bias_
  //           << "alpha scale" << scale_alpha << "beta scale" << scale_beta
  //           << std::endl;
  // int8_t* debug_input = reinterpret_cast<int8_t*>(malloc(m_ * k_));
  // std::cout << "input print" << std::endl;
  // cudaMemcpy(debug_input, x, m_ * k_, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 20; ++i) {
  //   for (int j = 0; j < k_; ++j) {
  //     std::cout << int(debug_input[i * k_ + j]) << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // free(debug_input);

  // int8_t* debug_input_convert = reinterpret_cast<int8_t*>(malloc(m_ * k_ /
  // 2)); std::cout << "input convert  print" << std::endl; cudaMemcpy(
  //     debug_input_convert, x_convert_, m_ * k_ / 2, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 20; ++i) {
  //   for (int j = 0; j < k_ / 2; ++j) {
  //     std::cout << std::hex << int(debug_input_convert[i * k_ / 2 + j]) << "
  //     ";
  //   }
  //   std::cout << std::dec << std::endl;
  // }
  // free(debug_input_convert);

  // int32_t* debug_weight = reinterpret_cast<int32_t*>(malloc(k_ * n_ * 4));
  // std::cout << "weight print" << std::endl;
  // cudaMemcpy(debug_weight, y_extra_, k_ * n_ * 4, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 20; ++i) {
  //   for (int j = 0; j < n_; ++j) {
  //     std::cout << debug_weight[i * n_ + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // free(debug_weight);

  // int32_t* debug_bias = reinterpret_cast<int32_t*>(malloc(m_ * n_ * 4));
  // std::cout << "bias print" << std::endl;
  // cudaMemcpy(debug_bias, bias_convert_, m_ * n_ * 4, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 20; ++i) {
  //   for (int j = 0; j < n_; ++j) {
  //     std::cout << debug_bias[i * n_ + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // free(debug_bias);

  // int32_t* debug = reinterpret_cast<int32_t*>(malloc(m_ * n_ * 4));
  // std::cout << "output print" << std::endl;
  // cudaMemcpy(debug, res_, m_ * n_ * 4, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 20; ++i) {
  //   for (int j = 0; j < n_; ++j) {
  //     std::cout << debug[i * n_ + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // free(debug);

  // int8_t* debug_res = reinterpret_cast<int8_t*>(malloc(m_ * n_));
  // std::cout << "output convert print" << std::endl;
  // cudaMemcpy(debug_res, res, m_ * n_, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 20; ++i) {
  //   for (int j = 0; j < n_; ++j) {
  //     std::cout << int(debug_res[i * n_ + j]) << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // free(debug_res);

  // auto* res = static_cast<__half*>(outputs[0]);
  // phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, __half>(
  //     res_, res, m_ * n_);
  // if (type_x_ == nvinfer1::DataType::kFLOAT) {
  //   float* res = static_cast<float*>(outputs[0]);
  //   phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, float>(
  //       res_, res, m_ * n_);
  // } else if (type_x_ == nvinfer1::DataType::kHALF) {
  //   auto res = static_cast<cutlass::half_t*>(outputs[0]);
  //   phi::fusion::cutlass_gemm_internal::ConvertData<int32_t,
  //   cutlass::half_t>(
  //       res_, res, m_ * n_);
  // } else if (type_x_ == nvinfer1::DataType::kINT8) {
  //   auto res = static_cast<int8_t*>(outputs[0]);
  //   phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int8_t>(
  //       res_, res, m_ * n_);
  // } else if (type_x_ == nvinfer1::DataType::kINT32) {
  //   auto res = static_cast<int32_t*>(outputs[0]);
  //   phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int32_t>(
  //       res_, res, m_ * n_);
  // }
  return cudaGetLastError() != cudaSuccess;
}

static nvinfer1::PluginFieldCollection field_collection_{0, nullptr};

MatmulInt4PluginCreator::MatmulInt4PluginCreator() {}

char const* MatmulInt4PluginCreator::getPluginName() const noexcept {
  return MATMULINT4PLUGINNAME;
}

char const* MatmulInt4PluginCreator::getPluginVersion() const noexcept {
  return MATMULINT4PLUGINVERSION;
}

const nvinfer1::PluginFieldCollection*
MatmulInt4PluginCreator::getFieldNames() noexcept {
  return &field_collection_;
}

void MatmulInt4PluginCreator::setPluginNamespace(
    char const* plugin_namespace) noexcept {
  plugin_namespace_ = plugin_namespace;
}

char const* MatmulInt4PluginCreator::getPluginNamespace() const noexcept {
  return plugin_namespace_.c_str();
}

float convertWeightFindScale(float* weight, size_t size, int range_size) {
  float w_min = std::abs(weight[0]), w_max = std::abs(weight[0]);
  for (size_t i = 1; i < size; ++i) {
    // w_min = std::min(w_min, weight[i]);
    w_max = std::max(w_max, std::abs(weight[i]));
  }
  float scale = w_max / range_size;
  for (size_t i = 0; i < size; ++i) {
    weight[i] /= scale;
  }
  return w_max;
}

float convertWeightFindScale(half* weight, size_t size, int range_size) {
  float w_min = std::abs(__half2float(weight[0])),
        w_max = std::abs(__half2float(weight[0]));
  for (size_t i = 1; i < size; ++i) {
    // w_min = std::min(w_min, __half2float(weight[i]));
    w_max = std::max(w_max, std::abs(__half2float(weight[i])));
  }
  float scale = w_max / range_size;
  for (size_t i = 0; i < size; ++i) {
    weight[i] = __float2half(__half2float(weight[i]) / scale);
  }
  return w_max;
}

// float convertWeightPerChannel(
//     float* weight, int m, int n, int column_major, int range_size) {
//   float* scales = new float[n];
//   for (int i = 0; i < n; ++i) {
//     int idx = column_major ? i * m : i;
//     scales[i] = std::abs(weight[idx]);
//   }
//   for (int i = 0; i < m * n; ++i) {
//     int idx = column_major ? i / m : i % n;
//     scales[idx] = std::max(scales[idx], std::abs(weight[i]));
//   }
//   for (int i = 0; i < n; ++i) {
//     scales[i] /= range_size;
//   }
//   for (int i = 0; i < m * n; ++i) {
//     int idx = column_major ? i / m : i % n;
//     weight[i] /= scales[idx];
//   }
//   float res = scales[0];
//   delete scales;
//   return res;
// }

nvinfer1::IPluginV2* MatmulInt4PluginCreator::createPlugin(
    char const* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  std::cout << "in create plugin" << std::endl;
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "before create plugin wrong"
              << " " << cudaGetErrorString(error) << std::endl;
  }
  nvinfer1::Dims dims_x, dims_y;
  nvinfer1::DataType x_type, y_type, bias_type;
  float scale_x, scale_y, scale_bias, scale_out;
  Int4GemmActivationType activation_type;
  bool with_bias = false, output_int4_range = false;
  void *y, *bias = nullptr;
  int m, n, k;
  for (int i = 0; i < fc->nbFields; ++i) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("dims_x") == 0) {
      dims_x = reinterpret_cast<const nvinfer1::Dims*>(fc->fields[i].data)[0];
      m = dims_x.d[dims_x.nbDims - 2];
    } else if (field_name.compare("dims_y") == 0) {
      dims_y = reinterpret_cast<const nvinfer1::Dims*>(fc->fields[i].data)[0];
      k = dims_y.d[dims_y.nbDims - 2];
      n = dims_y.d[dims_y.nbDims - 1];
    } else if (field_name.compare("type_x") == 0) {
      x_type =
          reinterpret_cast<const nvinfer1::DataType*>(fc->fields[i].data)[0];
    } else if (field_name.compare("type_y") == 0) {
      y_type =
          reinterpret_cast<const nvinfer1::DataType*>(fc->fields[i].data)[0];
    } else if (field_name.compare("type_bias") == 0) {
      bias_type =
          reinterpret_cast<const nvinfer1::DataType*>(fc->fields[i].data)[0];
    } else if (field_name.compare("scale_x") == 0) {
      scale_x = reinterpret_cast<const float*>(fc->fields[i].data)[0];
      // scale_x = scale_x / 7;
    } else if (field_name.compare("scale_out") == 0) {
      scale_out = reinterpret_cast<const float*>(fc->fields[i].data)[0];
      // scale_out = scale_out / 7;
    } else if (field_name.compare("activation_type") == 0) {
      activation_type = reinterpret_cast<const Int4GemmActivationType*>(
          fc->fields[i].data)[0];
    } else if (field_name.compare("output_int4_range") == 0) {
      output_int4_range =
          (reinterpret_cast<const int32_t*>(fc->fields[i].data)[0] != 0);
    } else if (field_name.compare("with_bias") == 0) {
      with_bias =
          (reinterpret_cast<const int32_t*>(fc->fields[i].data)[0] != 0);
    } else if (field_name.compare("y") == 0) {
      void* y_ori = const_cast<void*>(fc->fields[i].data);
      std::cout << "Y data size:" << fc->fields[i].length
                << " compute size:" << k * n << " byte size:" << sizeof(y_type)
                << "ytype" << int(y_type) << std::endl;
      float* debug = reinterpret_cast<float*>(y_ori);
      std::cout << "create plugin before covert weight print" << std::endl;
      // cudaMemcpy(debug, res, m_ * n_ * 4, cudaMemcpyDeviceToHost);
      for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 50; ++j) {
          std::cout << debug[i + j * k] << " ";
        }
        std::cout << std::endl;
      }
      // free(debug);
      if (y_type == nvinfer1::DataType::kHALF) {
        scale_y = convertWeightFindScale(
            reinterpret_cast<half*>(y_ori), fc->fields[i].length, 7);
      } else if (y_type == nvinfer1::DataType::kFLOAT) {
        scale_y = convertWeightFindScale(
            reinterpret_cast<float*>(y_ori), fc->fields[i].length, 7);
      }
      debug = reinterpret_cast<float*>(y_ori);
      std::cout << "create plugin after covert weight print" << std::endl;
      // cudaMemcpy(debug, res, m_ * n_ * 4, cudaMemcpyDeviceToHost);
      for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 50; ++j) {
          std::cout << debug[i + j * k] << " ";
        }
        std::cout << std::endl;
      }
      // free(debug);

      cudaMalloc(&y, k * n * sizeof(y_type));
      cudaMemcpy(y, y_ori, k * n * sizeof(y_type), cudaMemcpyHostToDevice);
    } else if (field_name.compare("bias") == 0) {
      void* bias_ori = const_cast<void*>(fc->fields[i].data);
      if (bias_type == nvinfer1::DataType::kFLOAT) {
        scale_bias = convertWeightFindScale(
            reinterpret_cast<float*>(bias_ori), fc->fields[i].length, 127);
      } else if (bias_type == nvinfer1::DataType::kHALF) {
        scale_bias = convertWeightFindScale(
            reinterpret_cast<half*>(bias_ori), fc->fields[i].length, 127);
      }
      // std::cout << "create plugin bias print" << std::endl;
      // // cudaMemcpy(debug, res, m_ * n_ * 4, cudaMemcpyDeviceToHost);
      // for (int j = 0; j < fc->fields[i].length; ++j) {
      //   std::cout << reinterpret_cast<float*>(bias_ori)[j] << " ";
      // }
      // std::cout << std::endl;
      // void* bias_device_ori;
      cudaMalloc(&bias, fc->fields[i].length * sizeof(bias_type));
      cudaMemcpy(bias,
                 bias_ori,
                 fc->fields[i].length * sizeof(bias_type),
                 cudaMemcpyHostToDevice);
      // cudaMalloc(&bias, m * n * sizeof(bias_type));
      // dim3 gridb(128);
      // dim3 blockb(768);
      // if (bias_type == nvinfer1::DataType::kFLOAT) {
      //   phi::fusion::cutlass_gemm_internal::ExpendKernel<float>
      //       <<<gridb, blockb>>>(reinterpret_cast<const
      //       float*>(bias_device_ori),
      //                           reinterpret_cast<float*>(bias),
      //                           n,
      //                           m,
      //                           0);
      // } else if (bias_type == nvinfer1::DataType::kINT32) {
      //   phi::fusion::cutlass_gemm_internal::ExpendKernel<int32_t>
      //       <<<gridb, blockb>>>(
      //           reinterpret_cast<const int32_t*>(bias_device_ori),
      //           reinterpret_cast<int32_t*>(bias),
      //           n,
      //           m,
      //           0);
      // } else if (bias_type == nvinfer1::DataType::kHALF) {
      //   phi::fusion::cutlass_gemm_internal::ExpendKernel<half>
      //       <<<gridb, blockb>>>(reinterpret_cast<const
      //       half*>(bias_device_ori),
      //                           reinterpret_cast<half*>(bias),
      //                           n,
      //                           m,
      //                           0);
      // }
      // cudaFree(bias_device_ori);
      std::cout << "bias expend m" << m << " n " << n << " bias type"
                << int32_t(bias_type) << " bias byte" << sizeof(bias_type)
                << std::endl;
    }
  }
  MatmulInt4Plugin* p = new MatmulInt4Plugin(dims_x,
                                             x_type,
                                             scale_x,
                                             dims_y,
                                             y_type,
                                             scale_y,
                                             activation_type,
                                             output_int4_range,
                                             with_bias,
                                             bias_type,
                                             scale_bias,
                                             scale_out,
                                             y,
                                             bias);
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "after create plugin wrong"
              << " " << cudaGetErrorString(error) << std::endl;
  }
  return p;
}

nvinfer1::IPluginV2* MatmulInt4PluginCreator::deserializePlugin(
    char const* name, void const* serial_data, size_t serial_length) noexcept {
  return new MatmulInt4Plugin(serial_data, serial_length);
}
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
