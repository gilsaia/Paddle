/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/convert/utils.h"
#include "paddle/fluid/inference/tensorrt/plugin/matmul_op_int4_plugin.h"

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {
namespace {
template <typename T>
void tranpose_weight(const T* src, T* dst, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[j * m + i] = src[i * n + j];
    }
  }
}
}  // namespace

/*
 * FC converter convert a MUL op in Fluid to a FC layer in TRT.
 */
class FcOpConverter : public OpConverter {
 public:
  void int4_plug(const framework::proto::OpDesc& op,
                 const framework::Scope& scope,
                 bool test_mode,
                 TensorRTEngine::Weight* weight,
                 TensorRTEngine::Weight* bias,
                 int m,
                 int n,
                 nvinfer1::ITensor* inputs,
                 const std::string& input_name,
                 const std::string& activation_type,
                 float in_scale,
                 float out_scale) {
    VLOG(3) << "convert a int4 matmul op to cutlass int4 plugin";
    std::cout << "in matmul int4 x scale" << in_scale << " out scale "
              << out_scale << std::endl;
    framework::OpDesc op_desc(op, nullptr);

    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input(input_name).front());
    // auto* input2 = engine_->GetITensor(op_desc.Input("W").front());

    nvinfer1::Dims dims_x = input1->getDimensions();
    // nvinfer1::Dims dims_y = input2->getDimensions();

    nvinfer1::DataType x_type = input1->getType();

    std::cout << "lambda x dim" << dims_x.nbDims;
    for (int i = 0; i < dims_x.nbDims; ++i) {
      std::cout << " " << dims_x.d[i] << " ";
    }
    std::cout << std::endl;

    // bool transpose_X = PADDLE_GET_CONST(bool,
    // op_desc.GetAttr("transpose_X"));
    // bool transpose_Y = false;
    // if (op_desc.HasAttr("transpose_Y")) {
    //   transpose_Y = PADDLE_GET_CONST(bool, op_desc.GetAttr("transpose_Y"));
    // }
    auto output_name = op_desc.Output("Out").front();

    // auto* input_convert_layer =
    //     TRT_ENGINE_ADD_LAYER(engine_, Identity, *input1);
    // input_convert_layer->setOutputType(0, nvinfer1::DataType::kINT32);

    std::vector<nvinfer1::ITensor*> plugin_inputs;
    // if (transpose_X) {
    //   nvinfer1::Permutation permutation;
    //   for (int i = 0; i < dims_x.nbDims - 2; ++i) {
    //     permutation.order[i] = i;
    //   }
    //   permutation.order[dims_x.nbDims - 2] = dims_x.nbDims - 1;
    //   permutation.order[dims_x.nbDims - 1] = dims_x.nbDims - 2;
    //   auto* transpose_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle,
    //   *input1); transpose_layer->setFirstTranspose(permutation);
    //   transpose_layer->setName(
    //       ("matmul_int4_op_transpose_x: Shuffle (Output:" + output_name +
    //       ")")
    //           .c_str());
    //   plugin_inputs.push_back(transpose_layer->getOutput(0));
    //   dims_x = plugin_inputs.back()->getDimensions();
    // } else {
    //   plugin_inputs.push_back(input1);
    // }
    plugin_inputs.push_back(input1);
    // if (!transpose_Y) {
    //   // cutlass int4 gemm need y in column major,so the action on y is
    //   opposite nvinfer1::Permutation permutation; for (int i = 0; i <
    //   dims_y.nbDims - 2; ++i) {
    //     permutation.order[i] = i;
    //   }
    //   permutation.order[dims_y.nbDims - 2] = dims_y.nbDims - 1;
    //   permutation.order[dims_y.nbDims - 1] = dims_y.nbDims - 2;
    //   auto* transpose_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle,
    //   *input2); transpose_layer->setFirstTranspose(permutation);
    //   transpose_layer->setName(
    //       ("matmul_int4_op_transpose_y: Shuffle (Output:" + output_name +
    //       ")")
    //           .c_str());
    //   plugin_inputs.push_back(transpose_layer->getOutput(0));
    // } else {
    //   plugin_inputs.push_back(input2);
    //   dims_y = plugin_inputs.back()->getDimensions();
    //   std::swap(dims_y.d[dims_y.nbDims - 1], dims_y.d[dims_y.nbDims - 2]);
    // }

    // nvinfer1::Dims dims_x_ = plugin_inputs[0]->getDimensions();
    // nvinfer1::Dims dims_y_ = plugin_inputs[1]->getDimensions();
    bool transpose_y = false;
    if (op_desc.HasAttr("transpose_Y")) {
      transpose_y = PADDLE_GET_CONST(bool, op_desc.GetAttr("transpose_Y"));
    }
    auto& weight_t = weight->get();
    nvinfer1::Dims dims_y;
    dims_y.nbDims = 2;
    if (!transpose_y) {
      dims_y.d[0] = weight->dims[1];
      dims_y.d[1] = weight->dims[0];
      // if (weight.get().type == nvinfer1::DataType::kFLOAT) {
      //   std::vector<float> weight_data_tmp;
      //   weight_data_tmp.reserve(weight_t.count);
      //   memcpy(weight_data_tmp.data(),
      //          weight.get().values,
      //          weight_t.count * sizeof(float));
      //   tranpose_weight(
      //       weight_data_tmp.data(),
      //       const_cast<float*>(static_cast<const
      //       float*>(weight.get().values)), dims_y.d[0], dims_y.d[1]);
      // } else if (weight.get().type == nvinfer1::DataType::kHALF) {
      //   std::vector<float16> weight_data_tmp;
      //   weight_data_tmp.reserve(weight_t.count);
      //   memcpy(weight_data_tmp.data(),
      //          weight.get().values,
      //          weight_t.count * sizeof(float16));
      //   tranpose_weight(weight_data_tmp.data(),
      //                   const_cast<float16*>(
      //                       static_cast<const
      //                       float16*>(weight.get().values)),
      //                   dims_y.d[0],
      //                   dims_y.d[1]);
      // }
    } else {
      dims_y.d[0] = weight->dims[0];
      dims_y.d[1] = weight->dims[1];
    }
    auto activation = paddle::inference::tensorrt::plugin::
        Int4GemmActivationType::INT4_GEMM_ACTIVATION_TYPE_NONE;
    bool with_bias = bias->get().values != nullptr;
    if (activation_type.compare("relu") == 0) {
      if (with_bias) {
        activation = paddle::inference::tensorrt::plugin::
            Int4GemmActivationType::INT4_GEMM_ACTIVATION_TYPE_BIAS_RELU;
      } else {
        activation = paddle::inference::tensorrt::plugin::
            Int4GemmActivationType::INT4_GEMM_ACTIVATION_TYPE_RELU;
      }
    } else {
      if (with_bias) {
        activation = paddle::inference::tensorrt::plugin::
            Int4GemmActivationType::INT4_GEMM_ACTIVATION_TYPE_BIAS;
      } else {
        activation = paddle::inference::tensorrt::plugin::
            Int4GemmActivationType::INT4_GEMM_ACTIVATION_TYPE_NONE;
      }
    }
    bool output_int4_range = (dims_y.d[1] == 3072);

    std::vector<nvinfer1::PluginField> fields;
    fields.emplace_back("dims_x", &dims_x, nvinfer1::PluginFieldType::kDIMS, 1);
    fields.emplace_back(
        "type_x", &x_type, nvinfer1::PluginFieldType::kINT32, 1);
    fields.emplace_back("dims_y", &dims_y, nvinfer1::PluginFieldType::kDIMS, 1);
    fields.emplace_back(
        "type_y", &weight_t.type, nvinfer1::PluginFieldType::kINT32, 1);
    fields.emplace_back(
        "activation_type", &activation, nvinfer1::PluginFieldType::kINT32, 1);
    fields.emplace_back("y",
                        weight_t.values,
                        GetPluginFieldType(weight_t.type),
                        weight_t.count);
    fields.emplace_back(
        "scale_x", &in_scale, nvinfer1::PluginFieldType::kFLOAT32, 1);
    fields.emplace_back(
        "scale_out", &out_scale, nvinfer1::PluginFieldType::kFLOAT32, 1);
    fields.emplace_back("output_int4_range",
                        &output_int4_range,
                        nvinfer1::PluginFieldType::kINT32,
                        1);
    if (with_bias) {
      fields.emplace_back(
          "with_bias", &with_bias, nvinfer1::PluginFieldType::kINT32, 1);
      fields.emplace_back(
          "type_bias", &bias->get().type, nvinfer1::PluginFieldType::kINT32, 1);
      fields.emplace_back("bias",
                          bias->get().values,
                          GetPluginFieldType(bias->get().type),
                          bias->get().count);
    }
    // fields.emplace_back("dims_y", &dims_y,
    // nvinfer1::PluginFieldType::kDIMS, 1);

    std::cout << "begin create plugin" << std::endl;

    nvinfer1::PluginFieldCollection* plugin_ptr =
        static_cast<nvinfer1::PluginFieldCollection*>(
            malloc(sizeof(*plugin_ptr) +
                   fields.size() * sizeof(nvinfer1::PluginField)));
    plugin_ptr->nbFields = fields.size();
    plugin_ptr->fields = fields.data();

    std::cout << "begin create plugin obj" << std::endl;

    std::cout << dims_x.nbDims << " " << dims_x.d[dims_x.nbDims - 1]
              << dims_x.d[dims_x.nbDims - 2] << std::endl;
    // std::cout << dims_y.nbDims << " " << dims_y.d[dims_y.nbDims - 1]
    //           << dims_y.d[dims_y.nbDims - 2] << std::endl;

    auto creator =
        GetPluginRegistry()->getPluginCreator("MatmulInt4Plugin", "1");
    std::cout << "create regis" << std::endl;
    auto plugin_obj = creator->createPlugin("MatmulInt4Plugin", plugin_ptr);
    std::cout << "create obj" << std::endl;
    std::cout << "with bias:" << with_bias << std::endl;
    auto* plugin_layer = engine_->network()->addPluginV2(
        plugin_inputs.data(), plugin_inputs.size(), *plugin_obj);

    std::cout << "end create plugin" << std::endl;
    // engine_->SetTensorDynamicRange(plugin_layer->getOutput(0), out_scale);
    if (output_int4_range) {
      engine_->SetTensorDynamicRange(plugin_layer->getOutput(0),
                                     out_scale / 7 * 127);
    } else {
      engine_->SetTensorDynamicRange(plugin_layer->getOutput(0), out_scale);
    }
    RreplenishLayerAndOutput(
        plugin_layer, "int4_gemm", {output_name}, test_mode);

    // if (with_bias) {
    //   nvinfer1::Dims bias_dim;
    //   bias_dim.nbDims = dims_x.nbDims;
    //   std::cout << "bias dims:" << bias_dim.nbDims << std::endl;
    //   for (int i = 0; i < (bias_dim.nbDims - 1); ++i) {
    //     bias_dim.d[i] = 1;
    //   }
    //   bias_dim.d[bias_dim.nbDims - 1] = bias->get().count;
    //   // bias_dim.d[bias_dim.nbDims - 1] = bias->dims.;
    //   std::cout << "init bias dims" << std::endl;
    //   auto* bias_tensor =
    //       TRT_ENGINE_ADD_LAYER(engine_, Constant, bias_dim, bias->get());
    //   engine_->SetTensorDynamicRange(bias_tensor->getOutput(0), out_scale);
    //   auto* add_layer =
    //       TRT_ENGINE_ADD_LAYER(engine_,
    //                            ElementWise,
    //                            *plugin_layer->getOutput(0),
    //                            *bias_tensor->getOutput(0),
    //                            nvinfer1::ElementWiseOperation::kSUM);
    //   add_layer->setOutputType(0, nvinfer1::DataType::kINT8);
    //   add_layer->setPrecision(nvinfer1::DataType::kINT8);
    //   RreplenishLayerAndOutput(
    //       add_layer, "after_int4_gemm_bias", {output_name}, test_mode);
    // } else {
    //   RreplenishLayerAndOutput(
    //       plugin_layer, "int4_gemm", {output_name}, test_mode);
    // }
    // plugin_layer->setName(
    //     ("matmul_int4: (Output: " + output_name + "_int32" + ")").c_str());
    // engine_->SetITensor(output_name + "_int32", plugin_layer->getOutput(0));

    // auto* identity_layer =
    //     TRT_ENGINE_ADD_LAYER(engine_, Identity, *plugin_layer->getOutput(0));
    // identity_layer->setOutputType(0, nvinfer1::DataType::kINT8);
    // engine_->SetTensorDynamicRange(identity_layer->getOutput(0), out_scale);
    // engine_->SetITensor(output_name, identity_layer->getOutput(0));

    // auto* iden_out_layer =
    //     TRT_ENGINE_ADD_LAYER(engine_, Identity,
    //     *identity_layer->getOutput(0));
    // engine_->SetITensor(output_name + ".debug",
    // iden_out_layer->getOutput(0)); engine_->DeclareOutput(output_name +
    // ".debug");

    free(plugin_ptr);
  }
  nvinfer1::ILayer* reshape_before_fc(nvinfer1::ITensor* before_fc,
                                      nvinfer1::Dims x_dim,
                                      int x_num_col_dims,
                                      std::string output_name) {
    // add shuffle before fc
    nvinfer1::Dims reshape_before_fc_dim;
    reshape_before_fc_dim.nbDims = x_num_col_dims + 3;
    // padding shape "* x q x 1 x 1"

    nvinfer1::ITensor* filal_reshape_before_fc_shape_tensor = nullptr;

    if (!engine_->with_dynamic_shape()) {
      for (int i = 0; i < reshape_before_fc_dim.nbDims; i++) {
        reshape_before_fc_dim.d[i] = 1;
      }
      for (int i = 0; i < x_dim.nbDims; i++) {
        if (i < x_num_col_dims) {
          reshape_before_fc_dim.d[i] = 0;
        } else {
          reshape_before_fc_dim.d[x_num_col_dims] *= x_dim.d[i];
        }
      }
    } else {
      std::vector<nvinfer1::ITensor*> reshape_before_fc_shape_tensor;
      nvinfer1::ITensor* input_shape_tensor = Shape(before_fc);

      for (int i = 0; i < reshape_before_fc_dim.nbDims; i++) {
        reshape_before_fc_shape_tensor.push_back(Add1DConstantLayer(1));
      }
      for (int i = 0; i < x_dim.nbDims; i++) {
        if (i < x_num_col_dims) {
          reshape_before_fc_shape_tensor[i] =
              GetEleTensorOfShape(input_shape_tensor, i);
        } else {
          reshape_before_fc_shape_tensor[x_num_col_dims] =
              Prod(GetEleTensorOfShape(input_shape_tensor, i),
                   reshape_before_fc_shape_tensor[x_num_col_dims]);
        }
      }
      filal_reshape_before_fc_shape_tensor =
          Concat(reshape_before_fc_shape_tensor);
    }

    auto* reshape_before_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *before_fc);
    if (!engine_->with_dynamic_shape()) {
      reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
    } else {
      reshape_before_fc_layer->setInput(1,
                                        *filal_reshape_before_fc_shape_tensor);
    }

    reshape_before_fc_layer->setName(
        ("fc_op_reshape_before_fc: Shuffle (Output: " + output_name + ")")
            .c_str());
    return reshape_before_fc_layer;
  }

  nvinfer1::ILayer* reshape_after_fc(nvinfer1::ITensor* after_fc,
                                     nvinfer1::Dims x_dim,
                                     int x_num_col_dims) {
    // add shuffle after fc
    nvinfer1::Dims reshape_after_fc_dim;
    reshape_after_fc_dim.nbDims = x_num_col_dims + 1;

    nvinfer1::ITensor* filal_reshape_after_fc_shape_tensor = nullptr;

    if (!engine_->with_dynamic_shape()) {
      for (int i = 0; i < reshape_after_fc_dim.nbDims; i++) {
        reshape_after_fc_dim.d[i] = 0;
      }
    } else {
      std::vector<int> gather_indices(x_num_col_dims + 1);
      std::iota(gather_indices.begin(), gather_indices.end(), 0);
      filal_reshape_after_fc_shape_tensor =
          Gather(Shape(after_fc), gather_indices);
    }

    auto* reshape_after_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *after_fc);
    if (!engine_->with_dynamic_shape()) {
      reshape_after_fc_layer->setReshapeDimensions(reshape_after_fc_dim);
    } else {
      reshape_after_fc_layer->setInput(1, *filal_reshape_after_fc_shape_tensor);
    }

    return reshape_after_fc_layer;
  }

  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid fc op to tensorrt fc layer without bias";
    framework::OpDesc op_desc(op, nullptr);
    auto output_name = op_desc.Output("Out").front();
    auto input_names = op_desc.InputNames();
    bool with_bias = input_names.size() >= 3;
    std::string w_name = "Y";
    std::string i_name = "X";
    if (with_bias) {
      w_name = "W";
      i_name = "Input";
    }
    // Declare inputs
    auto* X = engine_->GetITensor(op_desc.Input(i_name).front());
    auto x_dim = X->getDimensions();

    std::cout << "plugin init x dim" << x_dim.nbDims;
    for (int i = 0; i < x_dim.nbDims; ++i) {
      std::cout << " " << x_dim.d[i] << " ";
    }
    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input(w_name).front());
    PADDLE_ENFORCE_NOT_NULL(
        Y_v,
        platform::errors::NotFound(
            "Can not find %s presistale var of fc in scope.", w_name));
    auto* Y_t = Y_v->GetMutable<phi::DenseTensor>();
    int x_num_col_dims =
        op_desc.HasAttr("x_num_col_dims")
            ? PADDLE_GET_CONST(int, op_desc.GetAttr("x_num_col_dims"))
            : (op_desc.HasAttr("in_num_col_dims")
                   ? PADDLE_GET_CONST(int, op_desc.GetAttr("in_num_col_dims"))
                   : 1);
    const std::string activation_type =
        op_desc.HasAttr("activation_type")
            ? PADDLE_GET_CONST(std::string, op_desc.GetAttr("activation_type"))
            : "";

    bool enable_int8 = op_desc.HasAttr("enable_int8");
    bool support_int8 = false;
    if (op_desc.HasAttr("support_int8")) {
      support_int8 = PADDLE_GET_CONST(bool, op_desc.GetAttr("support_int8"));
    }
    float in_scale = 0;

    for (auto name : input_names) {
      std::cout << name << std::endl;
    }
    std::cout << "enable int8:" << enable_int8 << " support_int8"
              << support_int8 << std::endl;
    std::cout << "last" << std::endl;

    if (enable_int8 || support_int8) {
      if (enable_int8) {
        in_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Input_scale"));
      } else {
        in_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("X"));
      }
      engine_->SetTensorDynamicRange(X, in_scale);
    }

    PADDLE_ENFORCE_EQ(Y_t->dims().size(),
                      2UL,
                      platform::errors::InvalidArgument(
                          "The fc's weight should be a matrix with 2 dims, but "
                          "it's %d-dimensional.",
                          Y_t->dims().size()));  // a matrix
    int m = Y_t->dims()[0];
    int n = Y_t->dims()[1];

    auto regist_fc = [&](nvinfer1::ITensor* inputs,
                         int n_output,
                         TensorRTEngine::Weight& weight,
                         TensorRTEngine::Weight& bias) {
      std::cout << "in lambda" << std::endl;
      auto& dims_y = weight.dims;
      std::cout << "y dims " << dims_y[0] << " " << dims_y[1] << std::endl;
      // bool aligned = !(dims_y.front() % 8);
      bool need_int4 = (dims_y[0] == 3072 || dims_y[1] == 3072);
      if (need_int4 && (support_int8 || enable_int8)) {
        float out_scale = 0;
        if (enable_int8) {
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("out_threshold"),
              true,
              platform::errors::InvalidArgument(
                  "must have out threshold in fc layers in int8 mode"));
          out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        } else {
          out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Out"));
        }
        engine_->SetTensorDynamicRange(X, in_scale / 7 * 127);
        int4_plug(op,
                  scope,
                  test_mode,
                  &weight,
                  &bias,
                  m,
                  n,
                  inputs,
                  i_name,
                  activation_type,
                  in_scale,
                  out_scale);
      } else if (enable_int8 || support_int8) {
        // if (enable_int8 || support_int8) {
        // add conv layer
        float out_scale = 0;
        if (enable_int8) {
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("out_threshold"),
              true,
              platform::errors::InvalidArgument(
                  "must have out threshold in fc layers in int8 mode"));
          out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        } else {
          out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Out"));
        }
        nvinfer1::DimsHW nv_ksize(1, 1);
        auto* fc_layer_int8 = TRT_ENGINE_ADD_LAYER(engine_,
                                                   Convolution,
                                                   *inputs,
                                                   n_output,
                                                   nv_ksize,
                                                   weight.get(),
                                                   bias.get());
        fc_layer_int8->setName(
            ("fc_op_int8_conv1x1: Convolution (Output: " + output_name + ")")
                .c_str());
        engine_->SetTensorDynamicRange(fc_layer_int8->getOutput(0), out_scale);
        auto* fc_after_reshape_int8 = reshape_after_fc(
            fc_layer_int8->getOutput(0), x_dim, x_num_col_dims);
        if (activation_type == "relu") {
          fc_after_reshape_int8->setName(
              ("int8_reshape_after_fc: Shuffle (Output: " + output_name + ")")
                  .c_str());
          engine_->SetTensorDynamicRange(fc_after_reshape_int8->getOutput(0),
                                         out_scale);
          nvinfer1::IActivationLayer* relu_layer_int8 =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Activation,
                                   *(fc_after_reshape_int8->getOutput(0)),
                                   nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_int8,
                                   "relu_after_fc_shuffle",
                                   {output_name},
                                   test_mode);
        } else {
          RreplenishLayerAndOutput(fc_after_reshape_int8,
                                   "fc_op_int8_reshape_after_fc: Shuffle",
                                   {output_name},
                                   test_mode);
        }
      } else {
        // add fc layer
        auto* fc_layer_float = TRT_ENGINE_ADD_LAYER(engine_,
                                                    FullyConnected,
                                                    *inputs,
                                                    n_output,
                                                    weight.get(),
                                                    bias.get());
        fc_layer_float->setName(
            ("fc_op_float: FullyConnected (Output: " + output_name + ")")
                .c_str());
        auto* fc_after_reshape_float = reshape_after_fc(
            fc_layer_float->getOutput(0), x_dim, x_num_col_dims);
        if (activation_type == "relu") {
          fc_after_reshape_float->setName(
              ("float_reshape_after_fc: Shuffle (Output: " + output_name + ")")
                  .c_str());
          nvinfer1::IActivationLayer* relu_layer_float =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Activation,
                                   *(fc_after_reshape_float->getOutput(0)),
                                   nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_float,
                                   "relu_after_fc_shuffle",
                                   {output_name},
                                   test_mode);
        } else {
          RreplenishLayerAndOutput(fc_after_reshape_float,
                                   "shuffle_after_fc",
                                   {output_name},
                                   test_mode);
        }
      }
      auto out_dim = engine_->GetITensor(output_name)->getDimensions();
      std::cout << "after reshape fc shape:" << out_dim.nbDims;
      for (int i = 0; i < out_dim.nbDims; i++) {
        std::cout << " " << out_dim.d[i] << " ";
      }
      std::cout << std::endl;
    };

    bool transpose_y = false;
    if (op_desc.HasAttr("transpose_Y")) {
      transpose_y = PADDLE_GET_CONST(bool, op_desc.GetAttr("transpose_Y"));
    }
    int weight_w, weight_h;
    auto weight = engine_->GetTrtWeight(op_desc.Input(w_name).front(), *Y_t);

    std::cout << "Transpose Y:" << transpose_y << " With Bias:" << with_bias
              << std::endl;

    if (!transpose_y) {
      if (weight.get().type == nvinfer1::DataType::kFLOAT) {
        std::vector<float> weight_data_tmp;
        weight_data_tmp.reserve(Y_t->numel());
        memcpy(weight_data_tmp.data(),
               weight.get().values,
               Y_t->numel() * sizeof(float));
        tranpose_weight(
            weight_data_tmp.data(),
            const_cast<float*>(static_cast<const float*>(weight.get().values)),
            m,
            n);
      } else if (weight.get().type == nvinfer1::DataType::kHALF) {
        std::vector<float16> weight_data_tmp;
        weight_data_tmp.reserve(Y_t->numel());
        memcpy(weight_data_tmp.data(),
               weight.get().values,
               Y_t->numel() * sizeof(float16));
        tranpose_weight(weight_data_tmp.data(),
                        const_cast<float16*>(
                            static_cast<const float16*>(weight.get().values)),
                        m,
                        n);
      } else {
        PADDLE_THROW(paddle::platform::errors::InvalidArgument(
            "Paddle-TRT fc convert not supporte dtype, now only support fp32 "
            "and fp16."));
      }
      weight_w = n;
      weight_h = m;
    } else {
      weight_w = m;
      weight_h = n;
    }
    size_t n_output = weight_w;
    weight.dims.assign({weight_w, weight_h});

    TensorRTEngine::Weight bias{weight.get().type, nullptr, 0};
    if (with_bias) {
      auto* b_v = scope.GetVar(op_desc.Input("Bias").front());
      auto* b_t = b_v->GetMutable<phi::DenseTensor>();
      bias = engine_->GetTrtWeight(op_desc.Input("Bias").front(), *b_t);
    }

    // auto& bias_t = bias.get();
    // float* b_data =
    //     const_cast<float*>(reinterpret_cast<const float*>(bias_t.values));
    // for (int i = 0; i < bias_t.count; ++i) {
    //   b_data[i] = 0;
    // }

    // const float* weight_data =
    //     reinterpret_cast<const float*>(weight.get().values);
    // std::cout << "in converter print weight" << std::endl;
    // for (int i = 0; i < 20; ++i) {
    //   for (int j = 0; j < 50; ++j) {
    //     std::cout << weight_data[j + i * 3072] << " ";
    //   }
    //   std::cout << std::endl;
    // }

    // Running the TRT Static Shape mode: x_num_col_dims-1
    if (!engine_->with_dynamic_shape()) {
      x_num_col_dims--;
    }
    // If use tensorrt'oss, the x_dim and x_num_col_dims need change, and can
    // not add Shuffle layer in ernie's multihead.
    if (x_dim.nbDims == 4 && x_dim.d[2] == 1 && x_dim.d[3] == 1) {
      auto& dims_y = weight.dims;
      bool need_int4 = (dims_y[0] == 3072 || dims_y[1] == 3072);
      if (need_int4 && (support_int8 || enable_int8)) {
        float out_scale = 0;
        if (enable_int8) {
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("out_threshold"),
              true,
              platform::errors::InvalidArgument(
                  "must have out threshold in fc layers in int8 mode"));
          out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        } else {
          out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Out"));
        }
        engine_->SetTensorDynamicRange(X, in_scale / 7 * 127);
        int4_plug(op,
                  scope,
                  test_mode,
                  &weight,
                  &bias,
                  m,
                  n,
                  X,
                  i_name,
                  activation_type,
                  in_scale,
                  out_scale);
      } else if (enable_int8 || support_int8) {
        // if (enable_int8 || support_int8) {
        // add conv1x1 layer
        nvinfer1::DimsHW nv_ksize(1, 1);
        auto* fc_layer_int8 = TRT_ENGINE_ADD_LAYER(engine_,
                                                   Convolution,
                                                   *X,
                                                   n_output,
                                                   nv_ksize,
                                                   weight.get(),
                                                   bias.get());
        if (activation_type == "relu") {
          fc_layer_int8->setName(
              ("ernie_fc_op_int8: Convolution (Output: " + output_name + ")")
                  .c_str());
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("out_threshold"),
              true,
              platform::errors::InvalidArgument(
                  "must have out threshold in fc layers in int8 mode"));
          float out_scale = 0;
          if (enable_int8) {
            out_scale =
                PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
          } else {
            out_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Out"));
          }
          engine_->SetTensorDynamicRange(fc_layer_int8->getOutput(0),
                                         out_scale);
          nvinfer1::IActivationLayer* relu_layer_int8 =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Activation,
                                   *(fc_layer_int8->getOutput(0)),
                                   nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_int8,
                                   "relu_after_ernie_fc_int8",
                                   {output_name},
                                   test_mode);
        } else {
          RreplenishLayerAndOutput(fc_layer_int8,
                                   "ernie_fc_op_int8: Convolution",
                                   {output_name},
                                   test_mode);
        }
      } else {
        // add fc layer
        auto* fc_layer_float = TRT_ENGINE_ADD_LAYER(
            engine_, FullyConnected, *X, n_output, weight.get(), bias.get());
        if (activation_type == "relu") {
          fc_layer_float->setName(
              ("ernie_fc_op_float: (Output: " + output_name + ")").c_str());
          nvinfer1::IActivationLayer* relu_layer_float =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Activation,
                                   *(fc_layer_float->getOutput(0)),
                                   nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_float,
                                   "relu_after_ernie_fc_float",
                                   {output_name},
                                   test_mode);
        } else {
          RreplenishLayerAndOutput(
              fc_layer_float, "ernie_fc_op_float", {output_name}, test_mode);
        }
      }
    } else {  // need reshape input before and after fc
      PADDLE_ENFORCE_GT(
          x_dim.nbDims,
          x_num_col_dims,
          platform::errors::InvalidArgument(
              "Params and input dims mismatch. Paddle-TRT FC "
              "converter expects x_dim.nbDims > x_num_col_dims, but "
              "x_dim.nbDims : %d, x_num_col_dims : %d.",
              x_dim.nbDims,
              x_num_col_dims));
      auto* reshape_before_fc_layer =
          reshape_before_fc(X, x_dim, x_num_col_dims, output_name);
      auto* reshape_itensor = reshape_before_fc_layer->getOutput(0);
      if (enable_int8 || support_int8) {
        engine_->SetTensorDynamicRange(reshape_itensor, in_scale);
      }
      regist_fc(reshape_itensor, n_output, weight, bias);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fc, FcOpConverter);
