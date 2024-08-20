#pragma once

#include <cstdint>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "degine/proto/onnx.pb.h"

class ONNXSerializer {
public:
  ONNXSerializer(std::string fname) : fname_(std::move(fname)) {}

  bool Serialize(mlir::ModuleOp module) {
    std::error_code EC;
    llvm::ToolOutputFile out(fname_, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << "Error can not open file " << fname_
                   << ", message: " << EC.message() << '\n';
      return false;
    }
    degine::onnx::ModelProto model;
    if (!Translate(model, module)) {
      llvm::errs() << "Error translate mlir to onnx failed" << '\n';
      return false;
    }

    std::string buf;
    if (!model.SerializeToString(&buf)) {
      llvm::errs() << "Error can not serialize proto to buf" << '\n';
      return false;
    }
    out.os() << buf;
    out.keep();
    return true;
  }

private:
  bool Translate(degine::onnx::ModelProto &model, mlir::ModuleOp module) {
    for (auto &&op : module.getOps()) {
      bool flag =
          llvm::TypeSwitch<mlir::Operation *, bool>(&op)
              .Case([&](mlir::memref::GlobalOp op) {
                return Translate(model, op);
              })
              .Case([&](mlir::func::FuncOp op) { return Translate(model, op); })
              .Case(
                  [&](mlir::gpu::BinaryOp op) { return Translate(model, op); })
              .Default(false);
      if (!flag) {
        return false;
      }
    }
    return true;
  }

  bool Translate(degine::onnx::ModelProto &model, mlir::memref::GlobalOp op) {
    mlir::DenseIntOrFPElementsAttr attr =
        mlir::dyn_cast<mlir::DenseIntOrFPElementsAttr>(
            op.getInitialValueAttr());
    if (!attr) {
      return false;
    }
    bool flag =
        llvm::TypeSwitch<mlir::Type, bool>(attr.getElementType())
            .Case([&](mlir::Float32Type) {
              mlir::DenseFPElementsAttr f32_attr =
                  mlir::dyn_cast<mlir::DenseFPElementsAttr>(attr);
              if (!f32_attr) {
                return false;
              }

              mlir::RankedTensorType tensor_type =
                  mlir::dyn_cast<mlir::RankedTensorType>(f32_attr.getType());
              if (!tensor_type) {
                return false;
              }

              degine::onnx::NodeProto &node_pb =
                  *model.mutable_graph()->add_node();
              node_pb.set_name(op.getName());
              node_pb.add_output(op.getName());
              node_pb.set_op_type("Constant");
              degine::onnx::AttributeProto &attr_pb = *node_pb.add_attribute();
              attr_pb.set_type(degine::onnx::AttributeProto::AttributeType::
                                   AttributeProto_AttributeType_TENSOR);
              degine::onnx::TensorProto &tensor_pb = *attr_pb.mutable_t();
              for (auto dim : tensor_type.getShape()) {
                tensor_pb.add_dims(dim);
              }
              tensor_pb.set_data_type(degine::onnx::TensorProto::DataType::
                                          TensorProto_DataType_FLOAT);
              auto &float_data = *tensor_pb.mutable_float_data();
              float_data.Reserve(tensor_type.getNumElements());
              std::copy(f32_attr.value_begin<float>(),
                        f32_attr.value_end<float>(), float_data.begin());
              return true;
            })
            .Case([&](mlir::IntegerType dtype) {
              mlir::DenseIntElementsAttr int_attr =
                  mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr);
              if (!int_attr) {
                return false;
              }

              mlir::RankedTensorType tensor_type =
                  mlir::dyn_cast<mlir::RankedTensorType>(int_attr.getType());
              if (!tensor_type) {
                return false;
              }

              if (dtype.getIntOrFloatBitWidth() != 32 &&
                  dtype.getIntOrFloatBitWidth() != 64) {
                return false;
              }

              degine::onnx::NodeProto &node_pb =
                  *model.mutable_graph()->add_node();
              node_pb.set_name(op.getName());
              node_pb.add_output(op.getName());
              node_pb.set_op_type("Constant");
              degine::onnx::AttributeProto &attr_pb = *node_pb.add_attribute();
              attr_pb.set_type(degine::onnx::AttributeProto::AttributeType::
                                   AttributeProto_AttributeType_TENSOR);
              degine::onnx::TensorProto &tensor_pb = *attr_pb.mutable_t();
              for (auto dim : tensor_type.getShape()) {
                tensor_pb.add_dims(dim);
              }
              if (dtype.isSigned()) {
                if (dtype.getIntOrFloatBitWidth() == 32) {
                  tensor_pb.set_data_type(degine::onnx::TensorProto::DataType::
                                              TensorProto_DataType_INT32);
                  auto &int32_data = *tensor_pb.mutable_int32_data();
                  int32_data.Reserve(tensor_type.getNumElements());
                  std::copy(int_attr.value_begin<int>(),
                            int_attr.value_end<int>(), int32_data.begin());
                  return true;
                } else {
                  tensor_pb.set_data_type(degine::onnx::TensorProto::DataType::
                                              TensorProto_DataType_INT64);
                  auto &int64_data = *tensor_pb.mutable_int64_data();
                  int64_data.Reserve(tensor_type.getNumElements());
                  std::copy(int_attr.value_begin<std::int64_t>(),
                            int_attr.value_end<std::int64_t>(),
                            int64_data.begin());
                  return true;
                }
              } else {
                if (dtype.getIntOrFloatBitWidth() == 64) {
                  tensor_pb.set_data_type(degine::onnx::TensorProto::DataType::
                                              TensorProto_DataType_UINT64);
                  auto &uint64_data = *tensor_pb.mutable_uint64_data();
                  uint64_data.Reserve(tensor_type.getNumElements());
                  std::copy(int_attr.value_begin<std::uint64_t>(),
                            int_attr.value_end<std::uint64_t>(),
                            uint64_data.begin());
                }
              }

              return false;
            })
            .Default(false);
    return flag;
  }
  bool Translate(degine::onnx::ModelProto &model, mlir::func::FuncOp op) {
    // TODO: impl
    return false;
  }
  bool Translate(degine::onnx::ModelProto &model, mlir::gpu::BinaryOp op) {
    // TODO: impl
    return false;
  }

  std::string fname_;
};
