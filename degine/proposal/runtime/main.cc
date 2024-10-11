#include <cassert>
#include <fstream>

#include "fmt/base.h"

#include "CpuInferSession.h"
#include "GraphModule.h"
#include "degine/proposal/runtime/onnx.pb.h"

int main(int argc, char *argv[]) {
  onnx::ModelProto model_pb;
  std::ifstream ifs(argv[1], std::ios::binary);
  assert(ifs);
  assert(model_pb.ParseFromIstream(&ifs));

  GraphModule g(model_pb);
  CpuInferSession sess(g);

  std::vector<float> x(28 * 28, 1);

  sess.GetBuffer("x").buffer = x.data();
  sess.GetBuffer("y").buffer = x.data();
  sess.Infer();

  for (auto i = 0ul; i < x.size();) {
    for (auto j = 0ul; j < 16 && i < x.size(); j++, i++) {
      fmt::print("{} ", x[i]);
    }
    fmt::print("\n");
  }

  return 0;
}
