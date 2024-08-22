#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/transforms/Passes.h"
#include "torch-mlir/InitAll.h"

int main(int argc, char *argv[]) {
  mlir::registerAllPasses();

  mlir::stablehlo::registerPassPipelines();
  mlir::stablehlo::registerPasses();
  mlir::stablehlo::registerStablehloLinalgTransformsPasses();
  mlir::tosa::registerStablehloTOSATransformsPasses();

  mlir::torch::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);

  mlir::stablehlo::registerAllDialects(registry);

  mlir::torch::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}
