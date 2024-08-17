#include "mlir/IR/BuiltinOps.h"
#include "torch/csrc/jit/ir/ir.h"

#include "TorchToMLIR.h"

namespace degine {
namespace torch {

mlir::OwningOpRef<mlir::ModuleOp>
convertTorchToMLIR(mlir::MLIRContext &context,
                   const ::torch::jit::Module &jitModule) {
  // TODO: impl
  return nullptr;
}

} // namespace torch
} // namespace degine
