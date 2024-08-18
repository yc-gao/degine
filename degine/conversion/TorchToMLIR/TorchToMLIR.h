#include "mlir/IR/BuiltinOps.h"
#include "torch/csrc/jit/ir/ir.h"

namespace degine {

mlir::OwningOpRef<mlir::ModuleOp>
convertTorchToMLIR(mlir::MLIRContext &context,
                   const ::torch::jit::Module &jitModule);

} // namespace degine
