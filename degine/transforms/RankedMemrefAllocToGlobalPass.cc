#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

struct RankedMemAllocToGlobalRewriter
    : public mlir::OpRewritePattern<mlir::memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    return mlir::failure();
  }
};

} // namespace

namespace mlir {
namespace degine {

#define GEN_PASS_DEF_DEGINEMEMALLOCTOGLOBALPASS
#include "degine/transforms/Passes.h.inc"

struct RankedMemrefAllocToGlobalPass
    : public impl::DegineMemAllocToGlobalPassBase<
          RankedMemrefAllocToGlobalPass> {
  using DegineMemAllocToGlobalPassBase::DegineMemAllocToGlobalPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RankedMemAllocToGlobalRewriter>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace degine
} // namespace mlir
