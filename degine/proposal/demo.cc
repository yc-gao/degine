#include <fstream>
#include <stdexcept>

#include "degine/ir/onnx.pb.h"

#define DEGINE_THROW_IF(cond, msg)                                             \
  if (cond) {                                                                  \
    throw std::runtime_error(msg);                                             \
  }

class InferSession {
public:
  InferSession(const std::string &fname) {
    std::ifstream ifs(fname, std::ios::binary);
    DEGINE_THROW_IF(!ifs, "can not open model file")
    DEGINE_THROW_IF(!model_.ParseFromIstream(&ifs),
                    "can not parse model istream");
  }

private:
  onnx::ModelProto model_;
};

class CudaInferSession : public InferSession {
public:
  CudaInferSession(const std::string &fname) : InferSession(fname) {}
};

int main(int argc, char *argv[]) { return 0; }
