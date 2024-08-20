#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "torch-mlir/InitAll.h"
#include "torch/csrc/jit/serialization/import.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

#include "TorchToMLIR.h"
#include "class_annotator.h"
#include "import_options.h"

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::value_desc("filename"),
                                         llvm::cl::Required);
llvm::cl::opt<std::string> outputFilename("o", llvm::cl::value_desc("filename"),
                                          llvm::cl::Required);

void InitTorchConvertOptions(torch_mlir::ClassAnnotator &annotator,
                             torch_mlir::ImportOptions &options,
                             const torch::jit::Module &jitModule) {
  // TODO: impl
  annotator.exportNone(*jitModule.type());
  annotator.exportPath(*jitModule.type(), {"forward"});

  torch_mlir::ArgAnnotation argAnnotation;
  argAnnotation.shape = {1, 3, 224, 224};
  argAnnotation.dtype = c10::ScalarType::Float;
  argAnnotation.hasValueSemantics = true;
  annotator.annotateArgs(*jitModule.type(), {"forward"},
                         {torch_mlir::ArgAnnotation{}, argAnnotation});
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "degine importer");
  torch::jit::Module jitModule = torch::jit::load(inputFilename);
  torch_mlir::ClassAnnotator annotator;
  torch_mlir::ImportOptions options;
  InitTorchConvertOptions(annotator, options, jitModule);

  mlir::DialectRegistry registry;
  mlir::torch::registerAllDialects(registry);
  mlir::MLIRContext context(std::move(registry));
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      convertTorchToMLIR(context, jitModule, annotator, options);
  if (!module) {
    llvm::errs() << "Error convert torch to mlir failed\n";
    return 1;
  }

  std::error_code EC;
  llvm::ToolOutputFile out(outputFilename, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Error dump mlir failed, " << EC.message() << '\n';
    return 1;
  }
  out.os() << *module;
  out.keep();
  return 0;
}
