#pragma once

#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "fmt/format.h"

#include "degine/ir/onnx.pb.h"

class OperandInfo {
  static constexpr std::array<std::int64_t, 13> dtype2size{
      -1, // UNDEFINED
      sizeof(float),
      sizeof(std::uint8_t),
      sizeof(std::int8_t),
      sizeof(std::uint16_t),
      sizeof(std::int16_t),
      sizeof(std::int64_t),
      -1, // STRING,
      -1, // BOOL
      2,  // FLOAT16
      sizeof(double),
      sizeof(std::uint32_t),
      sizeof(std::uint64_t),
  };

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

  void Buffer(void *buf) { buffer = buf; }
  template <typename T = void> T *Buffer() {
    return reinterpret_cast<T *>(buffer);
  }

  std::string Name() const { return name; }

  std::int64_t ElemCount() const {
    return std::accumulate(elem_shape.begin(), elem_shape.end(), 1,
                           [](auto a, auto b) { return a * b; });
  }

  std::size_t ByteSize() const { return ElemCount() * dtype2size[elem_type]; }

private:
  std::string name;
  std::int32_t elem_type;
  std::vector<std::int64_t> elem_shape;
  void *buffer;
};

class OpInfo {
public:
  OpInfo(const onnx::NodeProto &node) : impl_(&node) {}
  OpInfo(const onnx::NodeProto *node) : impl_(node) {}

  std::string Name() const { return impl_->name(); }
  std::string OpType() const { return impl_->op_type(); }
  std::string Input(int idx) const { return impl_->input(idx); }
  std::string Output(int idx) const { return impl_->output(idx); }

private:
  const onnx::NodeProto *impl_;
};

using GraphInfo = onnx::GraphProto;
