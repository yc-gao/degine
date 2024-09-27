#include <cassert>
#include <fstream>

#include "degine/cpu/CpuInferSession.h"
#include "degine/ir/GraphModule.h"
#include "degine/ir/onnx.pb.h"

int main(int argc, char *argv[]) {
  onnx::ModelProto model_pb;
  std::ifstream ifs(argv[1], std::ios::binary);
  assert(ifs);
  assert(model_pb.ParseFromIstream(&ifs));

  GraphModule g(model_pb);
  CpuInferSession sess(g);
  sess.Infer();

  return 0;
}
