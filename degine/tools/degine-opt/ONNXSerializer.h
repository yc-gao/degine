#pragma once

#include <string>

#include "mlir/IR/BuiltinOps.h"

class ONNXSerializer {
public:
  ONNXSerializer(std::string fname) : fname_(std::move(fname)) {}

  bool Serialize(mlir::ModuleOp module) {
    // TODO: impl
    return false;
  }

private:
  std::string fname_;
};
