#include <memory>
#include <vector>

#include "degine/cpu/OpKernel.h"
#include "degine/ir/GraphModule.h"

class CpuInferSession {
public:
  CpuInferSession(const GraphModule &g) {}

  void Infer() {
    for (auto &&kernel : kernels_) {
      kernel->Infer();
    }
  }

private:
  std::vector<std::unique_ptr<OpKernel>> kernels_;
};
