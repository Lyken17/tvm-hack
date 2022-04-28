/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file convolution.cc
 * \brief Convolution operators
 */
#include "convolution.h"

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/reduction.h>

#include <vector>
#include <limits>
#include <numeric>

#include "../../transforms/infer_layout_utils.h"
#include "../op_common.h"
#include "convolution_make.h"

namespace tvm {
namespace relay {


// Newly added Mean OPs
    
// Newly added MCUConv2dOPs
inline Expr MakeMCUConv2D(Expr data, Expr weight, Expr bias,
                Expr zero_x, Expr zero_y, Expr effective_scale, 
                Array<IndexExpr> strides,
                Array<IndexExpr> padding, Array<IndexExpr> dilation,
                int groups, IndexExpr channels, Array<IndexExpr> kernel_size,
                std::string grad_layout, std::string data_layout,
                std::string kernel_layout, DataType out_dtype) 
{
  auto attrs = make_object<Conv2DAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->out_dtype = std::move(out_dtype);
  attrs->data_layout = std::move(grad_layout);
  attrs->kernel_layout = std::move(data_layout);
  attrs->out_layout = std::move(kernel_layout);
  const Op& op = Op::Get("nn.mcuconv2d");
  return Call(op, {data, weight, bias, zero_x, zero_y, effective_scale}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.mcuconv2d")
    .set_body_typed([](Expr data, Expr weight, Expr bias, Expr zero_x, Expr zero_y, Expr effective_scale, 
                       Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                       String out_layout, DataType out_dtype) {
      return MakeMCUConv2D(data, weight, bias,
                    zero_x, zero_y, effective_scale, 
                    strides, padding, dilation, groups, channels,
                    kernel_size, data_layout, kernel_layout, out_layout, out_dtype);
    });

bool MCUConv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 7) << "find num_inputs " << num_inputs << " expect types to be lenght {num_inputs + 1}";
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCHW("NCHW");
  Layout kOIHW("OIHW");

  const auto* param = attrs.as<Conv2DAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  bool is_dnnl_group_conv = false;
  if (param->groups > 1 && kernel_layout.name().find("G") != std::string::npos) {
    kOIHW = Layout("GOIHW");
    is_dnnl_group_conv = true;
  }

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCHW);
  if (!trans_in_layout.defined()) {
    reporter->GetDiagCtx().Emit(
        Diagnostic::Error(reporter->GetSpan())
        << "conv2d only support input layouts that are convertible from NCHW."
        << " The provided layout is: " << in_layout);
    return false;
  }

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIHW);
  if (!trans_kernel_layout.defined()) {
    reporter->GetDiagCtx().Emit(Diagnostic::Error(reporter->GetSpan())
                                << "conv2d only support kernel layouts that are convertible from "
                                << kOIHW << "."
                                << " The provided layout is: " << kernel_layout);
    return false;
  }

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCHW);
  if (!trans_out_layout.defined()) {
    reporter->GetDiagCtx().Emit(
        Diagnostic::Error(reporter->GetSpan())
        << "conv2d only support output layouts that are convertible from NCHW."
        << "The provided layout is: " << out_layout);
    return false;
  }

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);
  bool is_depthwise = false;
  if (param->groups > 1) {
    if (!(weight && weight->shape.defined())) {
      reporter->GetDiagCtx().Emit(
          Diagnostic::Error(reporter->GetSpan())
          << "Weight shape must be specified when groups is greater than 1.");
      return false;
    }

    Array<IndexExpr> wshape_oihw = trans_kernel_layout.ForwardShape(weight->shape);
    if (tvm::tir::ExprDeepEqual()(param->groups, dshape_nchw[1]) &&
        tvm::tir::ExprDeepEqual()(param->groups, wshape_oihw[0])) {
      is_depthwise = true;
    }
  }

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    ICHECK_EQ(param->kernel_size.size(), 2);
    ICHECK_EQ(param->dilation.size(), 2);
    Array<IndexExpr> wshape;

    if (is_dnnl_group_conv) {
      // infer weight's shape for group convolution
      wshape = {{param->groups, indexdiv(param->channels, param->groups),
                 indexdiv(dshape_nchw[1], param->groups), param->kernel_size[0],
                 param->kernel_size[1]}};
    } else if (is_depthwise) {
      // infer weight's shape for depthwise convolution
      wshape = {{dshape_nchw[1], indexdiv(param->channels, dshape_nchw[1]), param->kernel_size[0],
                 param->kernel_size[1]}};
    } else {
      wshape = {{param->channels, indexdiv(dshape_nchw[1], param->groups), param->kernel_size[0],
                 param->kernel_size[1]}};
    }

    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }

    if (param->auto_scheduler_rewritten_layout.size() == 0) {
      // Normal case: assign result to reporter
      reporter->Assign(types[1], TensorType(wshape, weight_dtype));
    } else {
      // If the layout is rewritten by auto-scheduler,
      // we just forcly apply the layout provided by auto-scheduler and
      // skip the normal inference logic.
      {}  // do nothing
    }
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;

    Array<PrimExpr> wshape;
    if (param->auto_scheduler_rewritten_layout.size() == 0) {
      wshape = weight->shape;
    } else {
      // works for the default kernel layout "HWIO"
      ICHECK_EQ(param->kernel_layout, "HWIO");
      wshape = auto_scheduler::GetShapeFromRewrittenLayout(param->auto_scheduler_rewritten_layout,
                                                           {"ry", "rx", "rc", "ff"});
    }

    wshape = trans_kernel_layout.ForwardShape(wshape);
    if (param->kernel_size.defined()) {
      ICHECK_EQ(param->kernel_size.size(), 2);

      if (!reporter->AssertEQ(param->kernel_size[0], wshape[2])) {
        reporter->GetDiagCtx().Emit(Diagnostic::Error(reporter->GetSpan())
                                    << "Conv2D: shape of weight is inconsistent with kernel_size,"
                                    << " kernel_size=" << param->kernel_size
                                    << " wshape=" << wshape);
      }

      if (!reporter->AssertEQ(param->kernel_size[1], wshape[3])) {
        reporter->GetDiagCtx().Emit(Diagnostic::Error(reporter->GetSpan())
                                    << "Conv2D: shape of weight is inconsistent with kernel_size,"
                                    << " kernel_size=" << param->kernel_size
                                    << " wshape=" << wshape);
        return false;
      }
    }

    if (param->channels.defined() && !reporter->AssertEQ(param->channels, wshape[0])) {
      reporter->GetDiagCtx().Emit(
          Diagnostic::Error(reporter->GetSpan())
          << "conv2D: the first dimensions of the weight tensor (" << wshape << ")"
          << "does not match the number of channels (" << param->channels << ").");
      return false;
    }

    if (!dshape_nchw[1].as<tir::AnyNode>() && !wshape[1].as<tir::AnyNode>()) {
      if (!reporter->AssertEQ(indexdiv(dshape_nchw[1], param->groups), wshape[1])) {
        reporter->GetDiagCtx().Emit(Diagnostic::Error(reporter->GetSpan())
                                    << "conv2d: requires that `"
                                    << indexdiv(dshape_nchw[1], param->groups) << "`,"
                                    << " the input channels (" << dshape_nchw[1] << ")"
                                    << " divided by groups (" << param->groups << ")"
                                    << ",\n must match the input channels"
                                    << " of the weight `" << wshape[1]
                                    << "`, where the weight shape is (" << wshape << ").");
        return false;
      }
    }
    channels = wshape[0];
    dilated_ksize_y = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});

  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  if (!dshape_nchw[2].as<tir::AnyNode>()) {
    oshape.Set(2, indexdiv(dshape_nchw[2] + pad_h - dilated_ksize_y, param->strides[0]) + 1);
  } else {
    oshape.Set(2, dshape_nchw[2]);
  }

  if (!dshape_nchw[3].as<tir::AnyNode>()) {
    oshape.Set(3, indexdiv(dshape_nchw[3] + pad_w - dilated_ksize_x, param->strides[1]) + 1);
  } else {
    oshape.Set(3, dshape_nchw[3]);
  }
  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  // std::cout << types << std::endl;
  reporter->Assign(types[num_inputs], TensorType(oshape, out_dtype));
  return true;
}


RELAY_REGISTER_OP("nn.mcuconv2d")
    .describe(R"code(test)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DAttrs>()
    .set_num_inputs(6)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .add_argument("bias", "Tensor", "The weight tensor.")
    .add_argument("zero_x", "Tensor", "The weight tensor.")
    .add_argument("zero_y", "Tensor", "The weight tensor.")
    .add_argument("effective_scale", "Tensor", "The weight tensor.")
    .set_support_level(2)
    .add_type_rel("MCUConv2D", MCUConv2DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv2DAttrs>);

// ================ MCU Elementwise Add
inline Expr MakeMCUAdd(Expr x1, Expr x2, 
                Expr zero_x1, Expr zero_x2, 
                Expr scale_x1, Expr scale_x2, 
                Expr zero_y, Expr scale_y,
                DataType out_dtype) 
{
  auto attrs = make_object<BiasAddAttrs>();
  const Op& op = Op::Get("nn.mcuadd");
  return Call(op, {x1, x2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.mcuadd")
    .set_body_typed([](Expr x1, Expr x2, 
                Expr zero_x1, Expr zero_x2, 
                Expr scale_x1, Expr scale_x2, 
                Expr zero_y, Expr scale_y,
                DataType out_dtype) {
      return MakeMCUAdd(x1, x2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y, out_dtype);
    });

bool MCUAddRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 9) << "find num_inputs " << num_inputs << " expect types to be lenght {num_inputs + 1}";
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  // assign output type
  reporter->Assign(types[num_inputs], TensorType(weight->shape, weight->dtype));
  return true;
}

RELAY_REGISTER_OP("nn.mcuadd")
    .describe(R"code(test)code" TVM_ADD_FILELINE)
    .set_attrs_type<BiasAddAttrs>()
    .set_num_inputs(8)
    .add_argument("x1", "Tensor", "The input tensor.")
    .add_argument("x2", "Tensor", "The weight tensor.")
    .add_argument("zero_x1", "Tensor", "The weight tensor.")
    .add_argument("zero_x2", "Tensor", "The weight tensor.")
    .add_argument("scale_x1", "Tensor", "The weight tensor.")
    .add_argument("scale_x2", "Tensor", "The weight tensor.")
    .add_argument("zero_y", "Tensor", "The weight tensor.")
    .add_argument("scale_y", "Tensor", "The weight tensor.")
    .set_support_level(2)
    .add_type_rel("MCUAdd", MCUAddRel);


}
}