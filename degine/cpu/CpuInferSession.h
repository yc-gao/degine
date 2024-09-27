#include <memory>
#include <vector>

#include "degine/cpu/OpKernel.h"
#include "degine/ir/GraphModule.h"

class CpuInferSession {
public:
  CpuInferSession(const GraphModule &g) {
    for (const OperandInfo *operand : g.Operands()) {
    }
    for (const OpInfo *op : g.Ops()) {
    }
  }

  void Infer() {
    for (auto &&kernel : kernels_) {
      kernel->Infer();
    }
  }

private:
  std::vector<std::unique_ptr<OpKernel>> kernels_;
};
