#include <fstream>

#include "fmt/core.h"

#include "degine/ir/onnx.pb.h"
#include "degine/proposal/InferSession.h"
#include "degine/proposal/kernel_registry.h"

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

  for (auto i = 0ul; i < x.size();) {
    for (int j = 0; j < 16 && i < x.size(); j++, i++) {
      fmt::print(" {} ", x[i]);
    }
    fmt::println("");
  }

  return 0;
}
