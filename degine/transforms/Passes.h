#pragma once

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace degine {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "degine/transforms/Passes.h.inc"

} // namespace degine
} // namespace mlir
