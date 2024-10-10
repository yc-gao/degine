#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace degine {

#define GEN_PASS_DEF_DEGINERANKEDTENSORBUFFERIZEPASS
#include "degine/transforms/Passes.h.inc"

struct RankedTensorBufferizePass
    : public impl::DegineRankedTensorBufferizePassBase<
          RankedTensorBufferizePass> {
  using DegineRankedTensorBufferizePassBase::
      DegineRankedTensorBufferizePassBase;

  void runOnOperation() override {}
};

} // namespace degine
} // namespace mlir
