#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "boost/preprocessor/cat.hpp"
#include "fmt/core.h"

#include "degine/proposal/graph.h"

class InferSession;
class OpKernel {
public:
  static constexpr std::int64_t kernel_id = -1;
  static constexpr std::int64_t priority = 100;

  static constexpr bool Match(const OpInfo &) { return true; }

  OpKernel(InferSession &, const OpInfo &) {}
  virtual void Infer() = 0;
};

class KernelRegistry {

  class KernelBuilder {
  public:
    virtual std::int64_t GetKernelId() const = 0;
    virtual std::int64_t GetPriority() const = 0;

    virtual bool Match(const OpInfo &) const = 0;

    virtual std::unique_ptr<OpKernel> BuildKernel(InferSession &,
                                                  const OpInfo &) = 0;

    bool operator<(const KernelBuilder &other) const {
      return GetPriority() < other.GetPriority();
    }
  };

  template <typename T> class KernelBuilderImpl : public KernelBuilder {
  public:
    std::int64_t GetKernelId() const override { return T::kernel_id; }
    std::int64_t GetPriority() const override { return T::priority; }
    bool Match(const OpInfo &op_info) const override {
      return T::Match(op_info);
    }

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
    }

    // kernel id match
    if (op_info.GetKernelId() != -1) {
      for (const auto &builder : iter->second) {
        if (builder->GetKernelId() == op_info.GetKernelId()) {
          return builder->BuildKernel(sess, op_info);
        }
      }
      return nullptr;
    }

    // common match
    for (const auto &builder : iter->second) {
      if (builder->Match(op_info)) {
        return builder->BuildKernel(sess, op_info);
      }
    }
    return nullptr;
  };

  template <typename T> void RegisterKernel(const std::string &optype) {
    auto &tmp = optype2builder_[optype];
    tmp.emplace_back(std::make_unique<KernelBuilderImpl<T>>());
    std::sort(tmp.begin(), tmp.end(),
              [](const auto &a, const auto &b) { return *a < *b; });
  }

private:
  std::unordered_map<std::string, std::vector<std::unique_ptr<KernelBuilder>>>
      optype2builder_;
};

#define DECLARE_OPKERNEL(optype, cls)                                          \
  namespace {                                                                  \
  bool BOOST_PP_CAT(flag, __COUNTER__) = []() {                                \
    KernelRegistry::Instance().RegisterKernel<cls>(optype);                    \
    return true;                                                               \
  }();                                                                         \
  }
