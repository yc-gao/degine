#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "torch/csrc/jit/api/module.h"

namespace {

bool translateTorchToMLIR(mlir::ModuleOp &op,
                          const torch::jit::ObjectPtr &iValue) {
  return false;
}

bool translateTorchToMLIR(mlir::ModuleOp &op,
                          const torch::jit::Module &jitModule) {
  return translateTorchToMLIR(op, jitModule._ivalue());
}

} // namespace

namespace torch_mlir {

mlir::OwningOpRef<mlir::ModuleOp>
convertTorchToMLIR(mlir::MLIRContext &context, torch::jit::Module jitModule) {
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
  if (translateTorchToMLIR(module, jitModule)) {
    return module;
  }
  return nullptr;
}

} // namespace torch_mlir
