#include <llvm/Support/CommandLine.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>

#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

namespace {

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::init("-"));

mlir::OwningOpRef<mlir::ModuleOp> LoadMLIR(mlir::MLIRContext &context,
                                           const std::string &fname) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(fname);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << fname << "\n";
    return nullptr;
  }

  return module;
}

void BuildGPuPipeline(mlir::PassManager &pm) {
  {
    // common passes
    pm.addPass(mlir::createConvertNVGPUToNVVMPass());
    pm.addPass(mlir::createGpuKernelOutliningPass());
    pm.addPass(mlir::createConvertVectorToSCFPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertNVVMToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());

    mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
    // nvvmTargetOptions.triple = options.cubinTriple;
    // nvvmTargetOptions.chip = options.cubinChip;
    // nvvmTargetOptions.features = options.cubinFeatures;
    // nvvmTargetOptions.optLevel = options.optLevel;
    pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    mlir::ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt;
    // convertIndexToLLVMPassOpt.indexBitwidth = options.indexBitWidth;
    pm.addPass(createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  {
    // gpu passes
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createStripDebugInfoPass());
    mlir::ConvertGpuOpsToNVVMOpsOptions opt;
    // opt.useBarePtrCallConv = options.kernelUseBarePtrCallConv;
    // opt.indexBitwidth = options.indexBitWidth;
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps(opt));
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(
        mlir::createReconcileUnrealizedCastsPass());
  }

  {
    // host passes
    mlir::GpuToLLVMConversionPassOptions opt;
    // opt.hostBarePtrCallConv = options.hostUseBarePtrCallConv;
    // opt.kernelBarePtrCallConv = options.kernelUseBarePtrCallConv;
    pm.addPass(createGpuToLLVMConversionPass(opt));

    mlir::GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
    // gpuModuleToBinaryPassOptions.compilationTarget = options.cubinFormat;
    pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }
}

} // namespace

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "mlir demo");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::MLIRContext context(std::move(registry));

  mlir::OwningOpRef<mlir::ModuleOp> module = LoadMLIR(context, inputFilename);
  if (!module) {
    return 1;
  }

  mlir::PassManager pm(module.get()->getName());
  BuildGPuPipeline(pm);
  if (mlir::failed(pm.run(*module))) {
    return 1;
  }
  module->dump();

  return 0;
}
