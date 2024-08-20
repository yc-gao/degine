#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "stablehlo/dialect/Register.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/InitAll.h"
#include "llvm/Support/CommandLine.h"

#include "ONNXSerializer.h"
#include "utils.h"

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::Required,
                                         llvm::cl::value_desc("filename"));
llvm::cl::opt<std::string> outputFilename("o", llvm::cl::init("output.onnx"));

void addPassesTorchScriptToLinalg(mlir::PassManager &pm) {
  mlir::torch::Torch::TorchLoweringPipelineOptions options;
  options.backendLegalOps = {"aten.flatten.using_ints",
                             "aten.adaptive_avg_pool1d"};
  mlir::torch::Torch::createTorchScriptModuleToTorchBackendPipeline(pm,
                                                                    options);
  mlir::torch::TorchConversion::
      createTorchBackendToLinalgOnTensorsBackendPipeline(pm);
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "degine optimizer");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::torch::registerAllDialects(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  mlir::MLIRContext context(std::move(registry));

  mlir::OwningOpRef<mlir::ModuleOp> module = LoadMLIR(context, inputFilename);
  if (!module) {
    llvm::errs() << "Error load mlir failed\n";
    return 1;
  }

  mlir::PassManager pm(module.get()->getName());
  addPassesTorchScriptToLinalg(pm);
  addPassesLinalgToGpu(pm);
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Error run PassManager failed\n";
    return 1;
  }

  ONNXSerializer serializer(outputFilename.getValue());
  if (!serializer.Serialize(*module)) {
    llvm::errs() << "Error serialize to onnx failed\n";
    return 1;
  }

  return 0;
}
