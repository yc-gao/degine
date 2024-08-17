#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/CommandLine.h"

#include "ONNXSerializer.h"
#include "degine/torch/TorchToMLIR.h"
#include "utils.h"

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::init("-"),
                                         llvm::cl::value_desc("filename"));
llvm::cl::opt<std::string> outputFilename("o", llvm::cl::init("output.onnx"));

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "degine optimizer");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  mlir::MLIRContext context(std::move(registry));

  auto module = LoadMLIR(context, inputFilename.getValue());
  if (!module) {
    return 1;
  }

  mlir::PassManager pm(module.get()->getName());
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
