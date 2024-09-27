#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "fmt/format.h"

#include "degine/cpu/OpKernel.h"
#include "degine/ir/GraphModule.h"

class CpuInferSession {
public:
  struct Buffer {
    void *buffer;
    std::size_t size;
  };

  CpuInferSession(const GraphModule &g) {
    for (const OperandInfo *operand : g.Operands()) {
      std::unique_ptr<char[]> buffer(new char[operand->ByteSize()]);

      auto ret = name2buffer_.emplace(
          operand->Name(), Buffer{buffer.get(), operand->ByteSize()});
      if (!ret.second) {
        throw std::runtime_error(
            fmt::format("can not index mem for operand {}", operand->Name()));
      }

      if (operand->Buffer()) {
        std::memcpy(buffer.get(), operand->Buffer(), operand->ByteSize());
      }
      buffers_.emplace_back(std::move(buffer));
    }

    for (const OpInfo *op : g.Ops()) {
      auto kernel =
          KernelRegistry<OpKernel, CpuInferSession, OpInfo>::Instance()
              .BuildKernel(*this, *op);
      if (!kernel) {
        throw std::runtime_error(fmt::format(
            "can not build kernel for op {}:{}", op->OpType(), op->Name()));
      }

      auto ret = name2kernel_.emplace(op->Name(), kernel.get());
      if (!ret.second) {
        throw std::runtime_error(fmt::format(
            "can not index kernel for op {}:{}", op->OpType(), op->Name()));
      }

      kernels_.emplace_back(std::move(kernel));
    }
  }

  Buffer &GetBuffer(const std::string &name) { return name2buffer_.at(name); }
  const Buffer &GetBuffer(const std::string &name) const {
    return name2buffer_.at(name);
  }
  void SetBuffer(const std::string &name, const Buffer &buf) {
    name2buffer_[name] = buf;
  }

  void Infer() {
    for (auto &&kernel : kernels_) {
      kernel->Infer();
    }
  }

private:
  std::vector<std::unique_ptr<char[]>> buffers_;
  std::vector<std::unique_ptr<OpKernel>> kernels_;

  std::unordered_map<std::string, Buffer> name2buffer_;
  std::unordered_map<std::string, OpKernel *> name2kernel_;
};
