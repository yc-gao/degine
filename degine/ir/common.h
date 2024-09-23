#pragma once

#include <cstdint>
#include <stdexcept>

#include "fmt/format.h"

#include "degine/ir/onnx.pb.h"

class OperandInfo {
public:
  OperandInfo(const onnx::ValueInfoProto &value_info) {
    name = value_info.name();

    const onnx::TypeProto &type_pb = value_info.type();
    if (!type_pb.has_tensor_type()) {
      throw std::runtime_error(
          fmt::format("undefined operand info, name {}", name));
    }
    const onnx::TypeProto::Tensor &tensor_type = type_pb.tensor_type();
    elem_type = tensor_type.elem_type();

    const onnx::TensorShapeProto &tensor_shape = tensor_type.shape();
    std::transform(tensor_shape.dim().begin(), tensor_shape.dim().end(),
                   std::back_inserter(elem_shape), [&](const auto &dim) {
                     if (!dim.has_dim_value()) {
                       throw std::runtime_error(
                           fmt::format("undefined shape dim, name {}", name));
                     }
                     return dim.dim_value();
                   });
  }

  OperandInfo(const onnx::TensorProto &tensor_pb) {
    name = tensor_pb.name();
    elem_type = tensor_pb.data_type();
    std::copy(tensor_pb.dims().begin(), tensor_pb.dims().end(),
              std::back_inserter(elem_shape));
  }

private:
  std::string name;
  std::int32_t elem_type;
  std::vector<std::int64_t> elem_shape;
};
class OpInfo {
public:
  OpInfo(const onnx::NodeProto &node) : impl_(&node) {}
  OpInfo(const onnx::NodeProto *node) : impl_(node) {}

  std::string OpType() const { return impl_->op_type(); }
  std::string Input(int idx) const { return impl_->input(idx); }
  std::string Output(int idx) const { return impl_->output(idx); }

private:
  const onnx::NodeProto *impl_;
};

using GraphInfo = onnx::GraphProto;
