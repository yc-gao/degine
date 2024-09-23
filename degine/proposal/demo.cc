#include <unordered_map>

#include "degine/ir/onnx.pb.h"
#include "degine/proposal/KernelRegistry.h"

class InferSession {
public:
  InferSession(const GraphInfo &graph_info) {}

  void Infer() {
    for (auto &&kernel : kernels_) {
      kernel->Infer(*this);
    }
  }

  OperandInfo *GetOperand(const std::string &name) {
    auto iter = name2operand_.find(name);
    if (iter != name2operand_.end()) {
      return &iter->second;
    }
    return nullptr;
  }

private:
  std::vector<std::unique_ptr<OpKernel>> kernels_;
  std::unordered_map<std::string, OperandInfo> name2operand_;
};

class AddKernel : public OpKernel {
public:
  AddKernel(const OpInfo &op_info) : OpKernel(op_info) {
    operand_a_ = op_info.Input(0);
    operand_b_ = op_info.Input(1);
    operand_c_ = op_info.Output(0);
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

int main(int argc, char *argv[]) {
  // onnx::TensorShapeProto tensor_shape;

  return 0;
}
