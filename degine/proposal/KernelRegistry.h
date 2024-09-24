#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "boost/preprocessor/cat.hpp"
#include "fmt/core.h"

#include "degine/proposal/graph.h"

class InferSession;
class OpKernel {
public:
  OpKernel(InferSession &, const OpInfo &) {}
  virtual void Infer() = 0;
};

class KernelRegistry {

  class KernelBuilder {
  public:
    virtual std::unique_ptr<OpKernel> BuildKernel(InferSession &,
                                                  const OpInfo &) = 0;
  };

  template <typename T> class KernelBuilderImpl : public KernelBuilder {
  public:
    std::unique_ptr<OpKernel> BuildKernel(InferSession &sess,
                                          const OpInfo &op_info) override {
      return std::make_unique<T>(sess, op_info);
    }
  };

public:
  static KernelRegistry &Instance() {
    static KernelRegistry inst;
    return inst;
  }

  std::unique_ptr<OpKernel> BuildKernel(InferSession &sess,
                                        const OpInfo &op_info) {
    auto iter = optype2builder_.find(op_info.OpType());
    if (iter == optype2builder_.end()) {
      return nullptr;
      // throw std::runtime_error(
      //     fmt::format("can not build kernel, optype {}", op_info.OpType()));
    }
    return iter->second->BuildKernel(sess, op_info);
  };

  template <typename T> void RegisterKernel(std::string optype) {
    if (!optype2builder_
             .emplace(std::move(optype),
                      std::make_unique<KernelBuilderImpl<T>>())
             .second) {
      throw std::runtime_error(
          fmt::format("can not register new kernel, optype {}", optype));
    }
  }

private:
  std::unordered_map<std::string, std::unique_ptr<KernelBuilder>>
      optype2builder_;
};

#define DECLARE_OPKERNEL(optype, cls)                                          \
  namespace {                                                                  \
  bool BOOST_PP_CAT(flag, __COUNTER__) = []() {                                \
    KernelRegistry::Instance().RegisterKernel<cls>(optype);                    \
    return true;                                                               \
  }();                                                                         \
  }
