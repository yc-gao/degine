#pragma once

#include <algorithm>
#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "boost/range/adaptor/transformed.hpp"
#include "boost/range/join.hpp"
#include "fmt/format.h"

#include "degine/proposal/runtime/onnx.pb.h"

class OperandInfo {
  static constexpr std::array dtype2size = {
      0ul,
      sizeof(float),         // FLOAT
      sizeof(std::uint16_t), // UINT8
      sizeof(std::int8_t),   // INT8
      sizeof(std::uint16_t), // UINT16
      sizeof(std::int16_t),  // INT16
      sizeof(std::int32_t),  // INT32
      sizeof(std::int64_t),  // INT64
      0ul,                   // STRING
      0ul,                   // BOOL
      0ul,                   // FLOAT16
      sizeof(double),        // DOUBLE
      sizeof(std::uint32_t), // UINT32
      sizeof(std::uint64_t), // UINT64
      0ul,                   // COMPLEX64
      0ul,                   // COMPLEX128
      0ul,                   // BFLOAT16
      0ul,                   // FLOAT8E4M3FN
      0ul,                   // FLOAT8E4M3FNUZ
      0ul,                   // FLOAT8E5M2
      0ul,                   // FLOAT8E5M2FNUZ
      0ul,                   // UINT4
      0ul,                   // INT4
      0ul,                   // FLOAT4E2M1
  };

public:
  enum DataType {
    UNDEFINED = onnx::TensorProto_DataType_UNDEFINED,
    FLOAT = onnx::TensorProto_DataType_FLOAT,
    UINT8 = onnx::TensorProto_DataType_UINT8,
    INT8 = onnx::TensorProto_DataType_INT8,
    UINT16 = onnx::TensorProto_DataType_UINT16,
    INT16 = onnx::TensorProto_DataType_INT16,
    INT32 = onnx::TensorProto_DataType_INT32,
    INT64 = onnx::TensorProto_DataType_INT64,
    STRING = onnx::TensorProto_DataType_STRING,
    BOOL = onnx::TensorProto_DataType_BOOL,
    FLOAT16 = onnx::TensorProto_DataType_FLOAT16,
    DOUBLE = onnx::TensorProto_DataType_DOUBLE,
    UINT32 = onnx::TensorProto_DataType_UINT32,
    UINT64 = onnx::TensorProto_DataType_UINT64,
    COMPLEX64 = onnx::TensorProto_DataType_COMPLEX64,
    COMPLEX128 = onnx::TensorProto_DataType_COMPLEX128,
    BFLOAT16 = onnx::TensorProto_DataType_BFLOAT16,
    FLOAT8E4M3FN = onnx::TensorProto_DataType_FLOAT8E4M3FN,
    FLOAT8E4M3FNUZ = onnx::TensorProto_DataType_FLOAT8E4M3FNUZ,
    FLOAT8E5M2 = onnx::TensorProto_DataType_FLOAT8E5M2,
    FLOAT8E5M2FNUZ = onnx::TensorProto_DataType_FLOAT8E5M2FNUZ,
    UINT4 = onnx::TensorProto_DataType_UINT4,
    INT4 = onnx::TensorProto_DataType_INT4,
    FLOAT4E2M1 = onnx::TensorProto_DataType_FLOAT4E2M1,
  };

  static OperandInfo FromOnnx(const onnx::ValueInfoProto &vinfo_pb) {
    OperandInfo operand;
    operand.name_ = vinfo_pb.name();

    auto &&type_pb = vinfo_pb.type();
    if (type_pb.has_tensor_type()) {
      auto &&tensor_type_pb = type_pb.tensor_type();
      operand.dtype_ = tensor_type_pb.elem_type();

      auto &&shape_pb = tensor_type_pb.shape();
      std::transform(shape_pb.dim().begin(), shape_pb.dim().end(),
                     std::back_inserter(operand.dims_), [](const auto &dim) {
                       if (dim.has_dim_value()) {
                         return dim.dim_value();
                       }
                       return 0l;
                     });
    } else {
      throw std::runtime_error(
          fmt::format("can not parse valueinfo, name {}", vinfo_pb.name()));
    }
    return operand;
  }
  static OperandInfo FromOnnx(const onnx::TensorProto &initializer_pb) {
    OperandInfo operand;
    operand.name_ = initializer_pb.name();
    operand.dtype_ = initializer_pb.data_type();
    operand.dims_ = std::vector<std::size_t>(initializer_pb.dims().begin(),
                                             initializer_pb.dims().end());
    operand.raw_buffer_ = std::unique_ptr<char[]>(new char[operand.ByteSize()]);
    switch (initializer_pb.data_type()) {
    case DataType::FLOAT: {
      std::memcpy(operand.raw_buffer_.get(), initializer_pb.float_data().data(),
                  operand.ByteSize());
    } break;
    case DataType::DOUBLE: {
      std::memcpy(operand.raw_buffer_.get(),
                  initializer_pb.double_data().data(), operand.ByteSize());
    } break;
    case DataType::INT32: {
      std::memcpy(operand.raw_buffer_.get(), initializer_pb.int32_data().data(),
                  operand.ByteSize());
    } break;
    case DataType::UINT32:
    case DataType::UINT64: {
      std::memcpy(operand.raw_buffer_.get(),
                  initializer_pb.uint64_data().data(), operand.ByteSize());
    } break;
    case DataType::INT64: {
      std::memcpy(operand.raw_buffer_.get(), initializer_pb.int64_data().data(),
                  operand.ByteSize());
    } break;
    default:
      throw std::runtime_error(
          fmt::format("can not parse TensorProto, name {} type{}",
                      initializer_pb.name(), int(initializer_pb.data_type())));
      break;
    }
    return operand;
  }

  std::string Name() const { return name_; }

  int Dtype() const { return dtype_; }

  std::size_t Dim(int idx) const { return dims_[idx]; }
  std::size_t DimCount() const { return dims_.size(); }

  void *Buffer() { return raw_buffer_.get(); }
  const void *Buffer() const { return raw_buffer_.get(); }

  std::size_t ElemCount() const {
    return std::accumulate(dims_.begin(), dims_.end(), 1,
                           std::multiplies<std::size_t>());
  }
  std::size_t ByteSize() const { return dtype2size[dtype_] * ElemCount(); }

private:
  std::string name_;

  int dtype_;
  std::vector<std::size_t> dims_;

  std::unique_ptr<char[]> raw_buffer_{nullptr};
};

class OpInfo {
public:
  static OpInfo FromOnnx(const onnx::NodeProto &node_pb,
                         std::vector<OperandInfo *> inputs,
                         std::vector<OperandInfo *> outputs) {
    OpInfo op;
    op.name_ = node_pb.name();
    op.optype_ = node_pb.op_type();
    op.inputs_ = std::move(inputs);
    op.outputs_ = std::move(outputs);
    for (const onnx::AttributeProto &attr : node_pb.attribute()) {
      switch (attr.type()) {
      case onnx::AttributeProto::INT: {
        op.attrs_.emplace(attr.name(), attr.i());
      } break;
      case onnx::AttributeProto::FLOAT: {
        op.attrs_.emplace(attr.name(), attr.f());
      } break;
      default:
        throw std::runtime_error(
            fmt::format("ca not parse node attr, name {} type {}", attr.name(),
                        int(attr.type())));
        break;
      }
    }
    return op;
  }
  std::string Name() const { return name_; }
  std::string OpType() const { return optype_; }

  const OperandInfo *Input(int idx) const { return inputs_[idx]; }
  OperandInfo *Input(int idx) { return inputs_[idx]; }
  std::size_t InputCount() const { return inputs_.size(); }

  const OperandInfo *Output(int idx) const { return outputs_[idx]; }
  OperandInfo *Output(int idx) { return outputs_[idx]; }
  std::size_t OutputCount() const { return outputs_.size(); }

  auto Attrs() const { return attrs_; }
  auto Attrs() { return attrs_; }

  template <typename T> const T &Attr(const std::string &name) const {
    return std::any_cast<const T &>(attrs_.at(name));
  }
  template <typename T> T &Attr(const std::string &name) {
    return std::any_cast<T &>(attrs_.at(name));
  }

  std::int64_t GetKernelId() const { return kid_; }

private:
  std::string name_;
  std::string optype_;

  std::vector<OperandInfo *> inputs_;
  std::vector<OperandInfo *> outputs_;

  std::unordered_map<std::string, std::any> attrs_;

  std::int64_t kid_{-1};
};

class GraphModule {
public:
  GraphModule(const onnx::ModelProto &model_pb) {
    const onnx::GraphProto &graph_pb = model_pb.graph();

    for (const onnx::TensorProto &initializer_pb : graph_pb.initializer()) {
      operands_.emplace_back(
          std::make_unique<OperandInfo>(OperandInfo::FromOnnx(initializer_pb)));
    }

    for (const onnx::ValueInfoProto &info_pb :
         boost::join(boost::join(graph_pb.input(), graph_pb.output()),
                     graph_pb.value_info())) {
      if (name2operand_.count(info_pb.name())) {
        // input, output, value_info may contains repeat
        continue;
      }
      operands_.emplace_back(
          std::make_unique<OperandInfo>(OperandInfo::FromOnnx(info_pb)));
      name2operand_[info_pb.name()] = operands_.back().get();
    }

    for (const onnx::NodeProto &node_pb : graph_pb.node()) {
      std::vector<OperandInfo *> inputs;
      std::vector<OperandInfo *> outputs;
      std::transform(node_pb.input().begin(), node_pb.input().end(),
                     std::back_inserter(inputs),
                     [this](const auto &k) { return name2operand_.at(k); });
      std::transform(node_pb.output().begin(), node_pb.output().end(),
                     std::back_inserter(outputs),
                     [this](const auto &k) { return name2operand_.at(k); });

      ops_.emplace_back(std::make_unique<OpInfo>(
          OpInfo::FromOnnx(node_pb, std::move(inputs), std::move(outputs))));

      name2op_[node_pb.name()] = ops_.back().get();
      for (OperandInfo *output : outputs) {
        operand2op_[output] = ops_.back().get();
      }
    }
  }

  auto Operands() const {
    return boost::make_iterator_range(operands_.begin(), operands_.end()) |
           boost::adaptors::transformed(
               [](const auto &ptr) { return ptr.get(); });
  }

  auto Ops() const {
    return boost::make_iterator_range(ops_.begin(), ops_.end()) |
           boost::adaptors::transformed(
               [](const auto &ptr) { return ptr.get(); });
  }

private:
  std::vector<std::unique_ptr<OperandInfo>> operands_;
  std::vector<std::unique_ptr<OpInfo>> ops_;

  // index for objects
  std::unordered_map<std::string, OperandInfo *> name2operand_;
  std::unordered_map<std::string, OpInfo *> name2op_;
  std::unordered_map<OperandInfo *, OpInfo *> operand2op_;
};
