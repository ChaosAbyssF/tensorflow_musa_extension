#ifndef TENSORFLOW_MUSA_MU1_DEVICE_MUSA_STREAM_H_
#define TENSORFLOW_MUSA_MU1_DEVICE_MUSA_STREAM_H_

#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/platform/port.h" // 确保引用了 port
#include <musa_runtime.h>

namespace stream_executor {
namespace musa {

// 我们不加 override，也不加 final
class MusaStream : public internal::StreamInterface {
 public:
  explicit MusaStream(musaStream_t stream) : musa_stream_(stream) {}
  ~MusaStream() override {}
  musaStream_t GetStream() const { return musa_stream_; }

  // ==============================================================
  // 【诊断模式】
  // 我们故意写一个名字稍微不一样的函数，或者去掉 override。
  // 这样编译器就会报错说“你没实现基类的 BlockHostUntilDone”。
  // 我们通过那个报错来看基类到底长什么样。
  // ==============================================================
  port::Status BlockHostUntilDone_DEBUG(Stream* stream) {
    musaError_t result = musaStreamSynchronize(musa_stream_);
    if (result != musaSuccess) {
         return port::Status(port::error::INTERNAL, "Sync Failed");
    }
    return port::Status::OK();
  }
  // ==============================================================

  void* GpuStreamHack() override { return (void*)musa_stream_; }
  void** GpuStreamMemberHack() override {
    return reinterpret_cast<void**>(&musa_stream_);
  }

 private:
  musaStream_t musa_stream_;
};

} // namespace musa
} // namespace stream_executor

#endif