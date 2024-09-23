#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include "degine/ir/onnx.pb.h"
#include "degine/proposal/KernelRegistry.h"

class InferSession {
public:
  InferSession(const GraphInfo &graph_info) {
    for (const auto &initializer : graph_info.initializer()) {
      auto ret =
          name2operand_.emplace(initializer.name(), OperandInfo(initializer));
      if (!ret.second) {
        throw std::runtime_error(fmt::format(
            "can not apply initializer, name {}", initializer.name()));
      }
      OperandInfo &operand = ret.first->second;
      std::unique_ptr<std::int8_t[]> buffer(
          new std::int8_t[operand.ByteSize()]);
      operand.Buffer(buffer.get());
      name2buffer_.emplace(operand.Name(), std::move(buffer));
      // TODO: impl
    }
    for (const auto &vinfo : graph_info.input()) {
      auto ret = name2operand_.emplace(vinfo.name(), OperandInfo(vinfo));
      if (ret.second) {
        OperandInfo &operand = ret.first->second;
        std::unique_ptr<std::int8_t[]> buffer(
            new std::int8_t[operand.ByteSize()]);
        operand.Buffer(buffer.get());
        name2buffer_.emplace(operand.Name(), std::move(buffer));
      }
    }
    for (const auto &vinfo : graph_info.output()) {
      auto ret = name2operand_.emplace(vinfo.name(), OperandInfo(vinfo));
      if (ret.second) {
        OperandInfo &operand = ret.first->second;
        std::unique_ptr<std::int8_t[]> buffer(
            new std::int8_t[operand.ByteSize()]);
        operand.Buffer(buffer.get());
        name2buffer_.emplace(operand.Name(), std::move(buffer));
      }
    }
    for (const auto &vinfo : graph_info.value_info()) {
      auto ret = name2operand_.emplace(vinfo.name(), OperandInfo(vinfo));
      if (ret.second) {
        OperandInfo &operand = ret.first->second;
        std::unique_ptr<std::int8_t[]> buffer(
            new std::int8_t[operand.ByteSize()]);
        operand.Buffer(buffer.get());
        name2buffer_.emplace(operand.Name(), std::move(buffer));
      }
    }
  }

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
  std::unordered_map<std::string, std::unique_ptr<std::int8_t[]>> name2buffer_;
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
  onnx::ModelProto model_pb;
  std::ifstream ifs(argv[1], std::ios::binary);
  model_pb.ParseFromIstream(&ifs);

  InferSession sess(model_pb.graph());

  std::vector<float> x(28 * 28);
  std::fill(x.begin(), x.end(), 1);

  sess.GetOperand("x")->Buffer(x.data());
  sess.GetOperand("y")->Buffer(x.data());
  sess.Infer();

  return 0;
}
