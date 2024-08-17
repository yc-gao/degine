#pragma once

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

inline mlir::OwningOpRef<mlir::ModuleOp>
LoadMLIR(mlir::MLIRContext &context, const std::string &inputFilename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "Error could not open input file " << ec.message() << '\n';
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << '\n';
    return nullptr;
  }
  return module;
}

inline void addPassesLinalgToParallelLoops(mlir::PassManager &pm) {
  // linalg to parallel loops
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

  pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
}

inline void addPassesParallepLoopsToGpu(mlir::PassManager &pm) {
  // parallel loops to gpu
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
  pm.addPass(mlir::createParallelLoopToGpuPass());
}

inline void addPassesLinalgToGpu(mlir::PassManager &pm) {
  addPassesLinalgToParallelLoops(pm);
  addPassesParallepLoopsToGpu(pm);

  // common passes
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createConvertSCFToCFPass());

  pm.addPass(mlir::createConvertNVGPUToNVVMPass());
  pm.addPass(mlir::createConvertNVVMToLLVMPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // gpu passes
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createGpuNVVMAttachTarget({}));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createStripDebugInfoPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createConvertGpuOpsToNVVMOps({}));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createGpuModuleToBinaryPass({}));
}
