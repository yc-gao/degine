#pragma once

#include <fstream>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#include "degine/proto/onnx.pb.h"

class ONNXSerializer {
public:
  ONNXSerializer(std::string fname) : fname_(std::move(fname)) {}

  bool Serialize(mlir::ModuleOp module) {
    std::ofstream ofs(fname_, std::ios::binary);
    if (!ofs.is_open()) {
      llvm::errs() << "Error can not open file" << '\n';
      return false;
    }
    degine::onnx::ModelProto model;
    if (!Translate(model, module)) {
      return false;
    }

    if (!model.SerializeToOstream(&ofs)) {
      llvm::errs() << "Error can not serialize to ostream" << '\n';
      return false;
    }
    return true;
  }

private:
  bool Translate(degine::onnx::ModelProto &model, mlir::ModuleOp module) {
    // TODO: impl
    return false;
  }

  std::string fname_;
};