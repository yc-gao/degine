#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "torch/csrc/jit/api/module.h"

namespace torch_mlir {

mlir::OwningOpRef<mlir::ModuleOp>
convertTorchToMLIR(mlir::MLIRContext &context, torch::jit::Module jitModule) {
  return nullptr;
}

} // namespace torch_mlir
