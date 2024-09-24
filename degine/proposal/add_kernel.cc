#include "degine/proposal/kernel.h"

struct AddKernel : public GeneralOpKernel<AddKernel> {
  using GeneralOpKernel::GeneralOpKernel;

  void Infer() override {
    float *x0 = Input(0)->Buffer<float>();
    float *x1 = Input(1)->Buffer<float>();
    float *y = Output(0)->Buffer<float>();

    for (int i = 0; i < Input(0)->ElemCount(); i++) {
      y[i] = x0[i] + x1[i];
    }
  }
};
DECLARE_OPKERNEL("Add", AddKernel)
