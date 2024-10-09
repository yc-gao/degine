#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

// struct OpKernel {
//   static constexpr std::int64_t kernel_id = -1;
//   static constexpr std::int64_t priority = 100;
//
//   static constexpr bool Match(const OpInfo &) { return true; }
//
//   OpKernel(InferSession &, const OpInfo &) {}
//   virtual void Infer() = 0;
// };

// struct OpInfo {
//   std::int64_t GetKernelId() const;
//   std::string OpType() const;
// };

template <typename OpKernel, typename InferSession, typename OpInfo>
class KernelRegistry {

  class OpBuilder {
  public:
    virtual std::int64_t GetKernelId() const = 0;
    virtual std::int64_t GetPriority() const = 0;

    virtual bool Match(const OpInfo &) const = 0;

    virtual std::unique_ptr<OpKernel> BuildKernel(InferSession &,
                                                  const OpInfo &) = 0;

    bool operator<(const OpBuilder &other) const {
      return GetPriority() < other.GetPriority();
    }
  };

  template <typename T> class OpBuilderImpl : public OpBuilder {
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

  template <typename T> void RegisterKernel(const std::string &optype) {
    auto &tmp = optype2builder_[optype];
    tmp.emplace_back(std::make_unique<OpBuilderImpl<T>>());
    std::sort(tmp.begin(), tmp.end(),
              [](const auto &a, const auto &b) { return *a < *b; });
  }
  std::unique_ptr<OpKernel> BuildKernel(InferSession &sess,
                                        const OpInfo &opinfo) {
    auto iter = optype2builder_.find(opinfo.OpType());
    if (iter == optype2builder_.end()) {
      return nullptr;
    }

    // kernel id match
    if (opinfo.GetKernelId() != -1) {
      for (const auto &builder : iter->second) {
        if (builder->GetKernelId() == opinfo.GetKernelId()) {
          return builder->BuildKernel(sess, opinfo);
        }
      }
      return nullptr;
    }

    // common match
    for (const auto &builder : iter->second) {
      if (builder->Match(opinfo)) {
        return builder->BuildKernel(sess, opinfo);
      }
    }
    return nullptr;
  }

private:
  std::unordered_map<std::string, std::vector<std::unique_ptr<OpBuilder>>>
      optype2builder_;
};
