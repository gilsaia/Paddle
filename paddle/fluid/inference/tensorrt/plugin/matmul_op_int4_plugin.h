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

#include <cuda.h>
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"

#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/inference/tensorrt/plugin/common/serialize.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/cuda_stream.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_decl.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_util.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
enum Int4GemmActivationType {
  INT4_GEMM_ACTIVATION_TYPE_NONE = 0,
  INT4_GEMM_ACTIVATION_TYPE_RELU = 1,
  INT4_GEMM_ACTIVATION_TYPE_BIAS = 2,
  INT4_GEMM_ACTIVATION_TYPE_BIAS_RELU = 3,
};
class MatmulInt4Plugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  //   MatmulInt4Plugin(nvinfer1::Dims const& dims_x,
  //                    nvinfer1::DataType type_x,
  //                    nvinfer1::Dims const& dims_y,
  //                    nvinfer1::DataType type_y,
  //                    Int4GemmActivationType activation_type,
  //                    void* y);
  MatmulInt4Plugin(nvinfer1::Dims const& dims_x,
                   nvinfer1::DataType type_x,
                   float scale_x,
                   nvinfer1::Dims const& dims_y,
                   nvinfer1::DataType type_y,
                   float scale_y,
                   Int4GemmActivationType activation_type,
                   bool with_bias,
                   nvinfer1::DataType type_bias,
                   float scale_bias,
                   float scale_out,
                   void* y,
                   void* bias);
  MatmulInt4Plugin(void const* data, size_t length);

  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int32_t output_index,
      nvinfer1::DimsExprs const* inputs,
      int32_t nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override;
  void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                       int32_t nb_inputs,
                       nvinfer1::DynamicPluginTensorDesc const* out,
                       int32_t nb_outputs) noexcept override;
  bool supportsFormatCombination(int32_t pos,
                                 nvinfer1::PluginTensorDesc const* in_out,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) noexcept override;
  size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* input_desc,
                          int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* output_desc,
                          int32_t nb_outputs) const noexcept override;
  int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                  nvinfer1::PluginTensorDesc const* output_desc,
                  void const* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(
      int32_t index,
      nvinfer1::DataType const* input_types,
      int32_t nb_inputs) const noexcept override;
  void attachToContext(cudnnContext* cudnnContext,
                       cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
  void detachFromContext() noexcept override;

  // IPluginV2 Methods
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int32_t getNbOutputs() const noexcept override;
  int32_t initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  char const* getPluginNamespace() const noexcept override;
  void setPluginNamespace(char const* plugin_name_space) noexcept override;

 protected:
  nvinfer1::Dims dims_x_;
  nvinfer1::Dims dims_y_;
  nvinfer1::DataType type_x_, type_y_, type_bias_;
  Int4GemmActivationType activation_type_;
  int batch_;
  uint64_t m_;
  uint64_t n_;
  uint64_t k_;
  float scale_x_, scale_y_, scale_bias_, scale_out_;
  int32_t *res_, *y_extra_, *bias_convert_, *x_extra_;
  cutlass::int4b_t *x_convert_, *y_convert_;
  //   void *y_ori_, *y_device_, *bias_ori_, *bias_device_;
  void *y_device_, *bias_device_;
  bool with_bias_;
  std::string namespace_;
};

class MatmulInt4PluginCreator : public nvinfer1::IPluginCreator {
 public:
  MatmulInt4PluginCreator();
  char const* getPluginName() const noexcept override;
  char const* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
  void setPluginNamespace(char const* plugin_namespace) noexcept override;
  char const* getPluginNamespace() const noexcept override;
  nvinfer1::IPluginV2* createPlugin(
      char const* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(
      char const* name,
      void const* serial_data,
      size_t serial_length) noexcept override;

 protected:
  std::string plugin_namespace_;
};

REGISTER_TRT_PLUGIN_V2(MatmulInt4PluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
