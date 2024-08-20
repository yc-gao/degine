#include "mlir/IR/BuiltinOps.h"
#include "torch/csrc/jit/ir/ir.h"

#include "class_annotator.h"
#include "import_options.h"

mlir::OwningOpRef<mlir::ModuleOp>
convertTorchToMLIR(mlir::MLIRContext &context,
                   const ::torch::jit::Module &jitModule,
                   torch_mlir::ClassAnnotator &annotator,
                   const torch_mlir::ImportOptions &options);
