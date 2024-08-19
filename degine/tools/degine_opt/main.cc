#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "stablehlo/dialect/Register.h"
#include "torch-mlir/InitAll.h"
#include "llvm/Support/CommandLine.h"

#include "ONNXSerializer.h"
#include "degine/conversion/TorchToMLIR/TorchToMLIR.h"
#include "utils.h"

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::init("-"),
                                         llvm::cl::value_desc("filename"));
enum Format {
  TORCH,
  MLIR,
};
llvm::cl::opt<enum Format> inputFormat(
    "f", llvm::cl::init(Format::MLIR),
    llvm::cl::values(clEnumValN(Format::TORCH, "torch", "torch module file")),
    llvm::cl::values(clEnumValN(Format::MLIR, "mlir", "mlir file")));

llvm::cl::opt<std::string> outputFilename("o", llvm::cl::init("output.onnx"));

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "degine optimizer");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::torch::registerAllDialects(registry);
  mlir::torch::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  mlir::MLIRContext context(std::move(registry));
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (inputFormat == Format::TORCH) {
    auto jitModule = LoadTorch(inputFilename);
    module = degine::convertTorchToMLIR(context, jitModule);
  } else if (inputFormat == Format::MLIR) {
    module = LoadMLIR(context, inputFilename);
  } else {
    llvm::errs() << "Error undefined input format" << '\n';
    return 1;
  }
  if (!module) {
    return 1;
  }

  mlir::PassManager pm(module.get()->getName());
  if (inputFormat == Format::TORCH) {
    degine::addPassesTorchToLinalg(pm);
  }
  addPassesLinalgToGpu(pm);
  if (mlir::failed(pm.run(*module))) {
    return 1;
  }

  ONNXSerializer serializer(outputFilename.getValue());
  if (!serializer.Serialize(*module)) {
    return 1;
  }

  return 0;
}
