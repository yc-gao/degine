#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "fmt/core.h"

#include "degine/ir/onnx.pb.h"

class InferSession;
class OpKernel {
public:
  virtual void Infer(InferSession &) = 0;
};

template <typename T = void> class KernelBuilder;
template <> class KernelBuilder<void> {
public:
  virtual std::unique_ptr<OpKernel>
  BuildKernel(const onnx::NodeProto &node) = 0;
};

template <typename T> class KernelBuilder : public KernelBuilder<void> {
public:
  virtual std::unique_ptr<OpKernel> BuildKernel(const onnx::NodeProto &node) {
    return std::make_unique<T>(node);
  }
};

class KernelRegistry {
public:
  static KernelRegistry &Instance() {
    static KernelRegistry inst;
    return inst;
  }

  std::unique_ptr<OpKernel> BuildKernel(const onnx::NodeProto &node) {
    auto iter = optype2builder_.find(node.op_type());
    if (iter != optype2builder_.end()) {
      return iter->second->BuildKernel(node);
    }
    return nullptr;
  };

  template <typename T> void RegisterKernel(std::string optype) {
    if (!optype2builder_
             .emplace(std::move(optype), std::make_unique<KernelBuilder<T>>())
             .second) {
      throw std::runtime_error(
          fmt::format("can not register new kernel {}", optype));
    }
  }

private:
  std::unordered_map<std::string, std::unique_ptr<KernelBuilder<>>>
      optype2builder_;
};

#define DECLARE_OPKERNEL(optype, cls)                                          \
  namespace {                                                                  \
  bool flag = []() {                                                           \
    KernelRegistry::Instance().RegisterKernel<cls>(optype);                    \
    return true;                                                               \
  }();                                                                         \
  }
