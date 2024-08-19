#include "TorchToMLIR.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch/csrc/jit/serialization/import.h"

#include "class_annotator.h"
#include "import_options.h"
#include "ivalue_importer.h"

namespace {

MlirStringRef toMlirStringRef(const char *s) {
  return mlirStringRefCreate(s, std::strlen(s));
}

MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

MlirModule createEmptyModule(MlirContext context) {
  MlirLocation loc = mlirLocationUnknownGet(context);
  return mlirModuleCreateEmpty(loc);
}

class ModuleBuilder {
public:
  ModuleBuilder(mlir::MLIRContext &context) : context(context) {}

  mlir::OwningOpRef<mlir::ModuleOp>
  importModule(const torch::jit::Module &jitModule) {
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module =
        builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

    torch_mlir::ClassAnnotator dummyAnnotator;
    torch_mlir::ClassAnnotator *classAnnotator = &dummyAnnotator;
    torch_mlir::ImportOptions importOptions;

    torch_mlir::importIValue(
        jitModule._ivalue(), mlirModuleGetBody(wrap(module)),
        mlirModuleGetContext(wrap(module)), *classAnnotator, importOptions);
    return module;
  }

private:
  mlir::MLIRContext &context;
};

} // namespace

namespace degine {

mlir::OwningOpRef<mlir::ModuleOp>
convertTorchToMLIR(mlir::MLIRContext &context,
                   const ::torch::jit::Module &jitModule) {
  return ModuleBuilder(context).importModule(jitModule);
}

void addPassesTorchToLinalg(mlir::PassManager &pm) {
  mlir::torch::Torch::TorchLoweringPipelineOptions options;
  options.backendLegalOps = {"aten.flatten.using_ints",
                             "aten.adaptive_avg_pool1d"};
  mlir::torch::Torch::createTorchScriptModuleToTorchBackendPipeline(pm,
                                                                    options);
  mlir::torch::TorchConversion::
      createTorchBackendToLinalgOnTensorsBackendPipeline(pm);
  // TODO: impl
}

} // namespace degine
