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
    std::string Name() const { return name_; }

    int Dtype() const { return dtype_; }

    std::size_t Dim(int idx) const { return dims_[idx]; }
    std::size_t DimCount() const { return dims_.size(); }

    std::size_t ElemCount() const {
      auto counter = boost::counting_range(0ul, DimCount());
      return std::transform_reduce(
          counter.begin(), counter.end(), 1,
          [this](const auto &a, const auto &b) { return a * b; },
          [this](const auto &idx) { return Dim(idx); });
    }

    template <typename T = void> T *Buffer() {
      return reinterpret_cast<T *>(Buffer<void>());
    }
    template <typename T = void> const T *Buffer() const {
      return reinterpret_cast<T *>(Buffer<void>());
    }
    std::size_t ByteSize() const;

    std::string name_;

    int dtype_;
    std::vector<std::size_t> dims_;

    void *buffer_;
  };

  class KernelInferCtx {
  public:
    Operand &Input(int idx) { return inputs_[idx]; }
    const Operand &Input(int idx) const { return inputs_[idx]; }
    std::size_t InputCount() const { return inputs_.size(); }

    Operand &Output(int idx) { return outputs_[idx]; }
    const Operand &Output(int idx) const { return outputs_[idx]; }
    std::size_t OutputCount() const { return outputs_.size(); }

    template <typename T> const T &Attr(const std::string &key) const {
      return std::any_cast<const T &>(attrs_.at(key));
    }

    friend class OpKernel;

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
