#include "boost/range/adaptor/transformed.hpp"
#include "boost/range/algorithm/copy.hpp"

#include "CpuInferSession.h"
#include "OpKernel.h"
#include "GraphModule.h"

OpKernel::OpKernel(CpuInferSession &sess, const OpInfo &opinfo) {
  for (auto s = 0ul, e = opinfo.InputCount(); s < e; s++) {
    auto i = opinfo.Input(s);

    Operand operand;
    operand.name_ = i->Name();
    operand.dtype_ = i->Dtype();
    boost::copy(
        boost::counting_range(0ul, i->DimCount()) |
            boost::adaptors::transformed([i](auto idx) { return i->Dim(idx); }),
        std::back_inserter(operand.dims_));
    operand.buffer_ = &sess.GetBuffer(i->Name());

    ctx_.inputs_.emplace_back(std::move(operand));
  }

  for (auto s = 0ul, e = opinfo.OutputCount(); s < e; s++) {
    auto o = opinfo.Output(s);

    Operand operand;
    operand.name_ = o->Name();
    operand.dtype_ = o->Dtype();
    boost::copy(
        boost::counting_range(0ul, o->DimCount()) |
            boost::adaptors::transformed([o](auto idx) { return o->Dim(idx); }),
        std::back_inserter(operand.dims_));
    operand.buffer_ = &sess.GetBuffer(o->Name());

    ctx_.outputs_.emplace_back(std::move(operand));
  }

  ctx_.attrs_ = opinfo.Attrs();
}

template <> void *OpKernel::Operand::Buffer() {
  return reinterpret_cast<CpuInferSession::Buffer *>(buffer_)->buffer;
}
template <> const void *OpKernel::Operand::Buffer() const {
  return reinterpret_cast<CpuInferSession::Buffer *>(buffer_)->buffer;
}

std::size_t OpKernel::Operand::ByteSize() const {
  return reinterpret_cast<CpuInferSession::Buffer *>(buffer_)->size;
}
