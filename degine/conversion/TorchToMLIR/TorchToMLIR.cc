#include "mlir/IR/BuiltinOps.h"
#include "torch/csrc/jit/ir/ir.h"

#include "TorchToMLIR.h"

namespace degine {

mlir::OwningOpRef<mlir::ModuleOp>
convertTorchToMLIR(mlir::MLIRContext &context,
                   const ::torch::jit::Module &jitModule) {
  jitModule.dump(true, false, false);
  // TODO: impl
  return nullptr;
}

} // namespace degine
