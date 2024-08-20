#include "TorchToMLIR.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "ivalue_importer.h"

mlir::OwningOpRef<mlir::ModuleOp>
convertTorchToMLIR(mlir::MLIRContext &context,
                   const ::torch::jit::Module &jitModule,
                   torch_mlir::ClassAnnotator &annotator,
                   const torch_mlir::ImportOptions &options) {
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module =
      builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  torch_mlir::importIValue(jitModule._ivalue(), mlirModuleGetBody(wrap(module)),
                           mlirModuleGetContext(wrap(module)), annotator,
                           options);
  return module;
}
