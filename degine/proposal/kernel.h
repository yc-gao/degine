#pragma once

#include "degine/proposal/InferSession.h"
#include "degine/proposal/kernel_registry.h"

template <typename Inhert> class GeneralOpKernel : public OpKernel {
public:
  GeneralOpKernel(InferSession &sess, const OpInfo &op_info)
      : OpKernel(sess, op_info) {
    for (auto i = 0ul, e = op_info.InputCount(); i < e; i++) {
      islots_.emplace_back(sess.GetOperand(op_info.Input(i)));
    }
    for (auto i = 0ul, e = op_info.OutputCount(); i < e; i++) {
      oslots_.emplace_back(sess.GetOperand(op_info.Output(i)));
    }
  }

  OperandInfo *Input(int idx) { return islots_[idx]; }
  OperandInfo *Output(int idx) { return oslots_[idx]; }

private:
  std::vector<OperandInfo *> islots_;
  std::vector<OperandInfo *> oslots_;
};
