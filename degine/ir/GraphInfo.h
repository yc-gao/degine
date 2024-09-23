#pragma once

#include "degine/ir/onnx.pb.h"

class OperandInfo {};
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
