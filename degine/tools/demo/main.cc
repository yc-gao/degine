#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/InitAll.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "ONNXSerializer.h"

llvm::cl::OptionCategory degineCategory("degine options");

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::value_desc("filename"),
                                         llvm::cl::cat(degineCategory));
llvm::cl::opt<std::string> outputFilename("o", llvm::cl::init("-"),
                                          llvm::cl::cat(degineCategory));

inline mlir::OwningOpRef<mlir::ModuleOp>
LoadMLIR(mlir::MLIRContext &context, const std::string &inputFilename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "Error can not open input file " << ec.message() << '\n';
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can not parse file " << inputFilename << '\n';
    return nullptr;
  }
  return module;
}

void addPassesTorchScriptLowering(mlir::PassManager &pm) {
  mlir::torch::Torch::TorchLoweringPipelineOptions options;
  options.backendLegalOps = {"aten.flatten.using_ints",
                             "aten.adaptive_avg_pool1d"};
  mlir::torch::Torch::createTorchScriptModuleToTorchBackendPipeline(pm,
                                                                    options);
  mlir::torch::TorchConversion::createTorchBackendToStablehloBackendPipeline(
      pm, {});
}

void addPassesStablehloLowering(mlir::PassManager &pm) {
  pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
}

void addPassesBufferize(mlir::PassManager &pm) {
  pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
  mlir::bufferization::OneShotBufferizationOptions opts;
  opts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferHoistingPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferLoopHoistingPass());
  pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createPromoteBuffersToStackPass());
  mlir::bufferization::buildBufferDeallocationPipeline(pm, {});
}

void addPassesLinalgLowering(mlir::PassManager &pm) {
  pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  pm.addPass(mlir::createParallelLoopFusionPass());
  pm.addPass(mlir::createParallelLoopTilingPass());
}

void addPassesParallelLoopLowering(mlir::PassManager &pm) {
  // Parallel Loops To GPu
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
  pm.addPass(mlir::createParallelLoopToGpuPass());

  // common passes
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // gpu passes
  pm.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuAsyncRegionPass());
  pm.addPass(mlir::createGpuNVVMAttachTarget({}));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createStripDebugInfoPass());

  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createConvertGpuOpsToNVVMOps({}));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createConvertNVGPUToNVVMPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertNVVMToLLVMPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createReconcileUnrealizedCastsPass());

  pm.addPass(mlir::createGpuModuleToBinaryPass({}));
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "degine optimizer");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);

  mlir::stablehlo::registerAllDialects(registry);

  mlir::torch::registerAllDialects(registry);

  mlir::MLIRContext context(std::move(registry));
  context.disableMultithreading();

  mlir::OwningOpRef<mlir::ModuleOp> module = LoadMLIR(context, inputFilename);
  if (!module) {
    llvm::errs() << "Error load mlir failed\n";
    return 1;
  }

  mlir::PassManager pm(module.get()->getName());
  pm.enableIRPrinting();
  addPassesTorchScriptLowering(pm);
  addPassesStablehloLowering(pm);
  addPassesBufferize(pm);
  addPassesLinalgLowering(pm);
  addPassesParallelLoopLowering(pm);
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Error run PassManager failed\n";
    return 1;
  }

  return 0;
}
