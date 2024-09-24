#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include "fmt/core.h"

#include "degine/ir/onnx.pb.h"
#include "degine/proposal/InferSession.h"
#include "degine/proposal/KernelRegistry.h"

template <typename Inhert> class GeneralOpKernel : public OpKernel {
public:
  GeneralOpKernel(InferSession &sess, const OpInfo &op_info)
      : OpKernel(sess, op_info) {
    for (auto i = 0ul, e = op_info.InputCount(); i < e; i++) {
      islots_.emplace_back(sess.GetOperand(op_info.Input(i)));
    }
    for (auto i = 0ul, e = op_info.OutputCount(); i < e; i++) {
      oslots_.emplace_back(sess.GetOperand(op_info.Output(i)));
    }
  }

  OperandInfo *Input(int idx) { return islots_[idx]; }
  OperandInfo *Output(int idx) { return oslots_[idx]; }

private:
  std::vector<OperandInfo *> islots_;
  std::vector<OperandInfo *> oslots_;
};

struct AddKernel : public GeneralOpKernel<AddKernel> {
  using GeneralOpKernel::GeneralOpKernel;

  void Infer() override {
    float *x0 = Input(0)->Buffer<float>();
    float *x1 = Input(1)->Buffer<float>();
    float *y = Output(0)->Buffer<float>();

    for (int i = 0; i < Input(0)->ElemCount(); i++) {
      y[i] = x0[i] + x1[i];
    }
  }
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
