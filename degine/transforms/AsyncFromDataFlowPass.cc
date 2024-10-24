#include <memory>

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"

namespace mlir {
namespace degine {

#define GEN_PASS_DEF_ASYNCFROMDATAFLOWPASS
#include "degine/transforms/Passes.h.inc"

struct AsyncFromDataFlowPass
    : public impl::AsyncFromDataFlowPassBase<AsyncFromDataFlowPass> {
  using AsyncFromDataFlowPassBase::AsyncFromDataFlowPassBase;

  void makeOpsAsync(mlir::Region &region) {

    llvm::SmallVector<mlir::Operation *, 8> worklist(llvm::make_pointer_range(
        llvm::make_filter_range(region.getOps(), [](mlir::Operation &op) {
          if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
            return false;
          }
          if (mlir::isa<mlir::async::ExecuteOp>(op)) {
            return false;
          }
          if (mlir::isa<mlir::async::AwaitOp>(op)) {
            return false;
          }
          return true;
        })));

    for (mlir::Operation *op : worklist) {
      mlir::OpBuilder builder(op);
      mlir::async::ExecuteOp execute_op =
          builder.create<mlir::async::ExecuteOp>(
              op->getLoc(), op->getResultTypes(), mlir::ValueRange{},
              mlir::ValueRange{});
      for (auto [async_val, val] :
           llvm::zip(execute_op.getBodyResults(), op->getResults())) {
        mlir::async::AwaitOp await_op =
            builder.create<mlir::async::AwaitOp>(op->getLoc(), async_val);
        val.replaceAllUsesWith(await_op.getResult());
      }
      op->moveBefore(execute_op.getBody(), execute_op.getBody()->end());
      builder.setInsertionPointToEnd(execute_op.getBody());
      builder.create<mlir::async::YieldOp>(op->getLoc(), op->getResults());
    }
  }

  void eliminateAwaitOp() {
    getOperation().walk([&](mlir::async::AwaitOp op) {
      mlir::async::ExecuteOp defined_op =
          op.getOperand().getDefiningOp<mlir::async::ExecuteOp>();
      bool all_execute = true;
      for (mlir::Operation *user : op.getResult().getUsers()) {
        mlir::async::ExecuteOp user_op =
            user->getParentOfType<mlir::async::ExecuteOp>();
        if (!user_op) {
          all_execute = false;
          continue;
        }
        user_op.getDependenciesMutable().append(defined_op.getToken());
        user_op.getBodyOperandsMutable().append(op.getOperand());
        user_op.getBody()->addArgument(op.getResult().getType(), op.getLoc());
        mlir::replaceAllUsesInRegionWith(
            op.getResult(), user_op.getBody()->getArguments().back(),
            user_op.getRegion());
      }
      if (all_execute) {
        op->erase();
      }
    });
  }

  void runOnOperation() override {
    makeOpsAsync(getOperation().getRegion());
    eliminateAwaitOp();
  }
};

} // namespace degine
} // namespace mlir
