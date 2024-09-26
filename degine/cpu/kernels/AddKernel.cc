#include "degine/cpu/OpKernel.h"

class AddKernel : public OpKernel {
public:
  using OpKernel::OpKernel;

  void Infer(KernelInferCtx &ctx) override {
    auto x0 = ctx.Input(0);
    auto x1 = ctx.Input(1);
    auto y0 = ctx.Output(0);
    for (auto s = 0ul, e = y0.ElemCount(); s < e; s++) {
      y0.Buffer<float>()[s] = x0.Buffer<float>()[s] + x1.Buffer<float>()[s];
    }
  }
};

DECLARE_OPKERNEL("Add", AddKernel)
