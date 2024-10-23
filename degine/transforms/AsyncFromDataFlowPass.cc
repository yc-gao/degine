#include <memory>

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace degine {

#define GEN_PASS_DEF_ASYNCFROMDATAFLOWPASS
#include "degine/transforms/Passes.h.inc"

struct AsyncFromDataFlowPass
    : public impl::AsyncFromDataFlowPassBase<AsyncFromDataFlowPass> {
  using AsyncFromDataFlowPassBase::AsyncFromDataFlowPassBase;

  void wrapOpUsingExecuteOp() {
    mlir::func::FuncOp func_op = getOperation();
    llvm::SmallVector<mlir::Operation *, 8> worklist;
    for (mlir::Operation &op : func_op.getOps()) {
      if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
        continue;
      }
      if (mlir::isa<mlir::async::ExecuteOp>(op)) {
        continue;
      }
      if (mlir::isa<mlir::async::AwaitOp>(op)) {
        continue;
      }
      worklist.push_back(&op);
    }

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

  void runOnOperation() override {
    wrapOpUsingExecuteOp();

    // mlir::func::FuncOp func_op = getOperation();
    // mlir::OpBuilder builder(func_op.getRegion());
    //
    // mlir::arith::ConstantOp const_op =
    //     builder.create<mlir::arith::ConstantIndexOp>(func_op.getLoc(), 0);
    //
    // mlir::async::ExecuteOp execute_op =
    // builder.create<mlir::async::ExecuteOp>(
    //     func_op.getLoc(), mlir::TypeRange{builder.getIndexType()},
    //     mlir::ValueRange{}, mlir::ValueRange{});
    // {
    //   mlir::OpBuilder::InsertionGuard guard(builder);
    //   builder.setInsertionPointToEnd(execute_op.getBody());
    //   mlir::arith::ConstantIndexOp constant_op =
    //       builder.create<mlir::arith::ConstantIndexOp>(func_op.getLoc(), 0);
    //   builder.create<mlir::async::YieldOp>(func_op.getLoc(),
    //                                        constant_op.getResult());
    // }
    //
    // mlir::async::ExecuteOp execute_op1 =
    // builder.create<mlir::async::ExecuteOp>(
    //     func_op.getLoc(), mlir::TypeRange{},
    //     mlir::ValueRange{execute_op.getToken()},
    //     execute_op.getBodyResults());
    // {
    //   mlir::OpBuilder::InsertionGuard guard(builder);
    //   builder.setInsertionPointToStart(execute_op1.getBody());
    //   builder.create<mlir::arith::AddIOp>(
    //       func_op.getLoc(), builder.getIndexType(),
    //       execute_op1.getBody()->getArgument(0), const_op.getResult());
    // }
  }
};

} // namespace degine
} // namespace mlir
