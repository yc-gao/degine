#include <memory>
#include <vector>

#include "degine/cpu/OpKernel.h"

class CpuInferSession {
public:
  void Infer() {
    for (auto &&kernel : kernels_) {
      kernel->Infer();
    }
  }

private:
  std::vector<std::unique_ptr<OpKernel>> kernels_;
};

int main(int argc, char *argv[]) {
  CpuInferSession sess;
  sess.Infer();
  return 0;
}
