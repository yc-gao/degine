#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace degine {

#define GEN_PASS_DEF_DEGINEDEMOPASS
#include "degine/transforms/Passes.h.inc"

struct DemoPass : public impl::DegineDemoPassBase<DemoPass> {
  using DegineDemoPassBase::DegineDemoPassBase;

  void runOnOperation() override {}
};

} // namespace degine
} // namespace mlir
