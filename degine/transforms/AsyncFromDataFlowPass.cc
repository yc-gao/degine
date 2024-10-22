#include <memory>

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace degine {

#define GEN_PASS_DEF_ASYNCFROMDATAFLOWPASS
#include "degine/transforms/Passes.h.inc"

struct AsyncFromDataFlowPass
    : public impl::AsyncFromDataFlowPassBase<AsyncFromDataFlowPass> {
  using AsyncFromDataFlowPassBase::AsyncFromDataFlowPassBase;

  void runOnOperation() override {}
};

} // namespace degine
} // namespace mlir
