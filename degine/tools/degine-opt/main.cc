#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/reference/InterpreterPasses.h"
#include "stablehlo/transforms/Passes.h"

#include "degine/tools/degine-opt/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::stablehlo::registerPassPipelines();
  mlir::stablehlo::registerPasses();
  mlir::stablehlo::registerStablehloLinalgTransformsPasses();
  mlir::stablehlo::registerInterpreterTransformsPasses();
  mlir::tosa::registerStablehloTOSATransformsPasses();

  mlir::degine::registerPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  mlir::stablehlo::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "degine modular optimizer driver\n", registry));
}
