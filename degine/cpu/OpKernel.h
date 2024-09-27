#pragma once

#include <any>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "boost/preprocessor/cat.hpp"
#include "boost/range/counting_range.hpp"

#include "degine/common/KernelRegistry.h"

class CpuInferSession;
class OpInfo;

class OpKernel {
public:
  static constexpr std::int64_t kernel_id = -1;
  static constexpr std::int64_t priority = 100;
  static constexpr bool Match(const OpInfo &) { return true; }

  OpKernel(CpuInferSession &, const OpInfo &);
  void Infer() { this->Infer(ctx_); }

protected:
  struct Operand {
    int DimCount() const { return dims_.size(); }
    std::int64_t Dim(int idx) const { return dims_[idx]; }

    std::size_t ElemCount() const {
      auto counter = boost::counting_range(0, DimCount());
      return std::transform_reduce(
          counter.begin(), counter.end(), 1,
          [this](const auto &a, const auto &b) { return a * b; },
          [this](const auto &idx) { return Dim(idx); });
    }

    template <typename T> T *Buffer() { return nullptr; }
    template <typename T> const T *Buffer() const { return nullptr; }

    std::string name_;
    std::vector<std::int64_t> dims_;
    void *buffer_;
  };

  class KernelInferCtx {
  public:
    const Operand Input(int idx) { return inputs_[idx]; }
    Operand Output(int idx) { return outputs_[idx]; }

    template <typename T> const T &Attr(const std::string &key) const {
      return std::any_cast<const T &>(attrs_.at(key));
    }

  private:
    std::vector<Operand> inputs_;
    std::vector<Operand> outputs_;

    std::unordered_map<std::string, std::any> attrs_;
  };
  virtual void Infer(KernelInferCtx &) = 0;

private:
  KernelInferCtx ctx_;
};

#define DECLARE_OPKERNEL(optype, cls)                                          \
  namespace {                                                                  \
  bool BOOST_PP_CAT(flag_, __COUNTER__) = []() {                               \
    KernelRegistry<::OpKernel, ::CpuInferSession, ::OpInfo>::Instance()        \
        .RegisterKernel<cls>(optype);                                          \
    return true;                                                               \
  }();                                                                         \
  }
