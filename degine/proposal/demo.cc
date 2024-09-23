#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include "fmt/core.h"

#include "degine/ir/onnx.pb.h"
#include "degine/proposal/KernelRegistry.h"

class InferSession {
  void InitBuffer(const GraphInfo &graph_info) {
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

  void InitOp(const GraphInfo &graph_info) {
    for (const OpInfo &op_info : graph_info.node()) {
      auto op = KernelRegistry::Instance().BuildKernel(*this, op_info);
      if (!op) {
        throw std::runtime_error(
            fmt::format("can not build kernel, optype {} opname {}",
                        op_info.OpType(), op_info.Name()));
      }
      kernels_.emplace_back(std::move(op));
    }
  }

public:
  InferSession(const GraphInfo &graph_info) {
    InitBuffer(graph_info);
    InitOp(graph_info);
  }

  void Infer() {
    for (auto &&kernel : kernels_) {
      kernel->Infer();
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
  AddKernel(InferSession &sess, const OpInfo &op_info)
      : OpKernel(sess, op_info) {
    opa = sess.GetOperand(op_info.Input(0));
    opb = sess.GetOperand(op_info.Input(1));
    opc = sess.GetOperand(op_info.Output(0));
  }

  void Infer() override {
    float *x0 = opa->Buffer<float>();
    float *x1 = opb->Buffer<float>();
    float *y = opc->Buffer<float>();

    for (int i = 0; i < opa->ElemCount(); i++) {
      y[i] = x0[i] + x1[i];
    }
  }

private:
  OperandInfo *opa;
  OperandInfo *opb;
  OperandInfo *opc;
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

  for (int i = 0; i < x.size();) {
    for (int j = 0; j < 16 && i < x.size(); j++, i++) {
      fmt::print(" {} ", x[i]);
    }
    fmt::println("");
  }

  return 0;
}
