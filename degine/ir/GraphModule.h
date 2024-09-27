#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <vector>

#include "boost/range/adaptor/transformed.hpp"
#include "boost/range/join.hpp"

#include "degine/ir/onnx.pb.h"

class OperandInfo {
public:
  static OperandInfo FromOnnx(const onnx::ValueInfoProto &vinfo_pb) {
    OperandInfo operand;
    return operand;
  }
  static OperandInfo FromOnnx(const onnx::TensorProto &initializer_pb) {
    OperandInfo operand;
    return operand;
  }

  std::string Name() const { return name_; }

  int Dtype() const { return dtype_; }

  std::size_t Dim(int idx) const { return dims_[idx]; }
  std::size_t DimCount() const { return dims_.size(); }

  std::size_t ByteSize() const { return 0; }

private:
  std::string name_;

  int dtype_;
  std::vector<std::size_t> dims_;

  std::unique_ptr<char[]> raw_buffer_;
};

class OpInfo {
public:
  static OpInfo FromOnnx(const onnx::NodeProto &node_pb,
                         std::vector<OperandInfo *> inputs,
                         std::vector<OperandInfo *> outputs) {
    OpInfo op;
    return op;
  }

  std::size_t InputCount() const { return 0; }
  const OperandInfo *Input(int idx) const { return nullptr; }

  std::size_t OutputCount() const { return 0; }
  const OperandInfo *Output(int idx) const { return nullptr; }

  std::string Name() const { return name_; }
  std::int64_t GetKernelId() const { return -1; }
  std::string OpType() const { return optype_; }

private:
  std::string name_;
  std::string optype_;
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
