#ifndef TENSORFLOW_MUSA_MU_DEVICE_MUSA_EXECUTOR_H_
#define TENSORFLOW_MUSA_MU_DEVICE_MUSA_EXECUTOR_H_

#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include <memory>

// 【新增】引用你的功能头文件
#include "musa_stream.h" 
#include "musa_event.h"  
#include "musa_memcpy.h"  // 你的 Memcpy 定义
#include "musa_memset.h"  // 你的 Memset 定义
#include "musa_device.h"
namespace stream_executor {
namespace musa {

// 辅助函数：将你的 mStatus 转换为 TF 的 port::Status
// 假设 mStatus::SUCCESS 是 0 (或者枚举的第一个)
inline port::Status FromMusaStatus(mStatus s) {
    if (s == mStatus::SUCCESS) { // 请确认你的 SUCCESS 定义
        return port::Status::OK();
    }
    return port::Status(port::error::INTERNAL, "MUSA Operation Failed");
}

class MusaExecutor : public internal::StreamExecutorInterface {
 public:
  explicit MusaExecutor(const PluginConfig& plugin_config) : plugin_config_(plugin_config) {}
  ~MusaExecutor() override {}

  port::Status Init(int device_ordinal, DeviceOptions device_options) override {
    device_ordinal_ = device_ordinal;
    return port::Status::OK();
  }

  // ========================================================================
  // 1. 工厂接口
  // ========================================================================
  
  std::unique_ptr<internal::StreamInterface> GetStreamImplementation() override {
      musaStream_t h;
      musaStreamCreate(&h);
      return std::make_unique<MusaStream>(h);
  }

  std::unique_ptr<internal::EventInterface> CreateEventImplementation() override {
    return std::make_unique<MusaEvent>();
  }

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation() override {
      return nullptr; 
  }

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override {
      return nullptr; 
  }

  // ========================================================================
  // 2. 内存管理接口
  // ========================================================================

  DeviceMemoryBase Allocate(uint64 size, int64 memory_space) override { 
      // 这里通常需要调用 musaMalloc，如果你有 musa_allocator.h 也可以在这里用
      // 暂时保留空实现，通常 DeviceContext 会处理分配
      return DeviceMemoryBase(nullptr, 0); 
  }
  
  void* GetSubBuffer(DeviceMemoryBase* parent, uint64 offset, uint64 size) override { 
      return reinterpret_cast<char*>(parent->opaque()) + offset;
  }

  void Deallocate(DeviceMemoryBase* mem) override {
      // musaFree(mem->opaque()); 
  }

  bool HostMemoryRegister(void* mem, uint64 size) override { return true; }
  bool HostMemoryUnregister(void* mem) override { return true; }

  void* HostMemoryAllocate(uint64 size) override { return nullptr; }
  void HostMemoryDeallocate(void* mem) override {}

  // ========================================================================
  // 3. 内存拷贝 (同步) - 填坑完毕！
  // ========================================================================
  
  port::Status SynchronousMemZero(DeviceMemoryBase* location, uint64 size) override { 
      // Memset 同步版，这里临时创建一个 handle
      mHandle h; 
      // 注意：这里没有 stream，默认走 0 流或默认流
      return FromMusaStatus(tensorflow::musa::Memset(h, location->opaque(), size, 0));
  }

  port::Status SynchronousMemSet(DeviceMemoryBase* location, int value, uint64 size) override { 
      mHandle h;
      // Memset 只支持 uint8 pattern
      return FromMusaStatus(tensorflow::musa::Memset(h, location->opaque(), size, static_cast<uint8_t>(value)));
  }

  port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst, const void* host_src, uint64 size) override { 
      // H2D
      return FromMusaStatus(tensorflow::musa::MusaMemcpyH2D(gpu_dst->opaque(), host_src, size));
  }

  port::Status SynchronousMemcpy(void* host_dst, const DeviceMemoryBase& gpu_src, uint64 size) override { 
      // D2H
      return FromMusaStatus(tensorflow::musa::MusaMemcpyD2H(host_dst, gpu_src.opaque(), size));
  }

  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64 size) override { 
      // D2D
      return FromMusaStatus(tensorflow::musa::MusaMemcpyD2D(gpu_dst->opaque(), gpu_src.opaque(), size));
  }
  
  // ========================================================================
  // 4. 内存拷贝 (异步) - 填坑完毕！
  // ========================================================================

  // 辅助：从 TF Stream 中获取底层 musaStream_t
  musaStream_t GetMusaStream(Stream* stream) {
      auto* musa_stream_impl = static_cast<MusaStream*>(stream->implementation());
      // 【注意】这里假设你在 musa_stream.h 里加了 GetStream() 方法
      return musa_stream_impl->GetStream(); 
  }

  // D2D Async
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64 size) override { 
      auto status = tensorflow::musa::MusaMemcpyAsyncD2D(
          gpu_dst->opaque(), gpu_src.opaque(), size, GetMusaStream(stream));
      return status == mStatus::SUCCESS;
  }

  // H2D Async
  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src, uint64 size) override { 
      auto status = tensorflow::musa::MusaMemcpyAsyncH2D(
          gpu_dst->opaque(), host_src, size, GetMusaStream(stream));
      return status == mStatus::SUCCESS;
  }

  // D2H Async
  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src, uint64 size) override { 
      auto status = tensorflow::musa::MusaMemcpyAsyncD2H(
          host_dst, gpu_src.opaque(), size, GetMusaStream(stream));
      return status == mStatus::SUCCESS;
  }

  // MemZero Async
  port::Status MemZero(Stream* stream, DeviceMemoryBase* location, uint64 size) override { 
      mHandle h;
      h.SetStream(GetMusaStream(stream)); // 绑定流
      return FromMusaStatus(tensorflow::musa::Memset(h, location->opaque(), size, 0));
  }

  // Memset32 Async
  port::Status Memset32(Stream* stream, DeviceMemoryBase* location, uint32 pattern, uint64 size) override { 
      mHandle h;
      h.SetStream(GetMusaStream(stream)); // 绑定流
      return FromMusaStatus(tensorflow::musa::Memset32(h, location->opaque(), size, pattern));
  }
  
  // ========================================================================
  // 5. 其他接口
  // ========================================================================

  // 这里的 BlockHostUntilDone 保持上次修复后的样子（带参数）
  port::Status BlockHostUntilDone(Stream* stream) override {
      internal::StreamInterface* implementation = stream->implementation();
      auto* musa_stream = static_cast<MusaStream*>(implementation);
      return musa_stream->BlockHostUntilDone_DEBUG(stream);
  }

  bool HostCallback(Stream* stream, std::function<port::Status()> callback) override { 
      // 如果 MUSA 支持 host callback，可以在这里实现
      return true; 
  }
  
  bool AllocateTimer(Timer* timer) override { return true; }
  void DeallocateTimer(Timer* timer) override {}
  bool StartTimer(Stream* stream, Timer* timer) override { return true; }
  bool StopTimer(Stream* stream, Timer* timer) override { return true; }

  int PlatformDeviceCount() override { return 1; }
  port::Status EnablePeerAccessTo(StreamExecutorInterface* other) override { return port::Status::OK(); }
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override { return false; }

  port::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription() const override {
      internal::DeviceDescriptionBuilder builder;
      builder.set_name("MUSA Device");
      return builder.Build();
  }

  bool SynchronizeAllActivity() override { return true; }
  bool DeviceMemoryUsage(int64* free, int64* total) const override { return false; }
  bool AllocateStream(Stream* stream) override { return true; }
  void DeallocateStream(Stream* stream) override {}
  bool CreateStreamDependency(Stream* dependent, Stream* other) override { return true; }

  port::Status AllocateEvent(Event* event) override { return port::Status::OK(); }
  port::Status DeallocateEvent(Event* event) override { return port::Status::OK(); }
  port::Status RecordEvent(Stream* stream, Event* event) override { return port::Status::OK(); }
  port::Status WaitForEvent(Stream* stream, Event* event) override { return port::Status::OK(); }
  Event::Status PollForEventStatus(Event* event) override { return Event::Status::kComplete; }

 private:
  PluginConfig plugin_config_;
  int device_ordinal_;
};

} // namespace musa
} // namespace stream_executor

#endif
