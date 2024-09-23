#include <unordered_map>

#include "degine/ir/onnx.pb.h"
#include "degine/proposal/KernelRegistry.h"

class OperandInfo {
public:
};

class InferSession {
public:
  OperandInfo *GetOperand(const std::string &name) {
    auto iter = name2operand_.find(name);
    if (iter != name2operand_.end()) {
      return &iter->second;
    }
    return nullptr;
  }

private:
  std::unordered_map<std::string, OperandInfo> name2operand_;
};

class AddKernel : public OpKernel {
public:
  AddKernel(const onnx::NodeProto &node) {
    operand_a_ = node.input(0);
    operand_b_ = node.input(1);
    operand_c_ = node.output(0);
  }

  void Infer(InferSession &session) override {
    OperandInfo *a = session.GetOperand(operand_a_);
    OperandInfo *b = session.GetOperand(operand_b_);
    OperandInfo *c = session.GetOperand(operand_c_);
  }

private:
  std::string operand_a_;
  std::string operand_b_;
  std::string operand_c_;
};
DECLARE_OPKERNEL("Add", AddKernel)

int main(int argc, char *argv[]) { return 0; }
