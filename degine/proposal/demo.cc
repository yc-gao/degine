#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include "fmt/core.h"

#include "degine/ir/onnx.pb.h"
#include "degine/proposal/InferSession.h"
#include "degine/proposal/KernelRegistry.h"

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
