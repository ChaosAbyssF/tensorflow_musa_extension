/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include "utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/platform/mutex.h"
#include <unordered_map>

namespace tensorflow {
namespace musa {

// 1. 定义一个最简单的资源包装器，避开 LookupInterface 的 ABI 坑
template <typename K, typename V>
class MusaTableResource : public ResourceBase {
 public:
  MusaTableResource() {}

  string DebugString() const override { return "MusaTableResource"; }

  mutex mu;
  std::unordered_map<K, V> data;
};

// 2. 实现创建 Table 的 Op
template <typename K, typename V>
class MusaMutableHashTableOp : public MusaOpKernel {
 public:
  explicit MusaMutableHashTableOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
    node_name_ = ctx->def().name();
  }

  void Compute(OpKernelContext* ctx) override {
    auto r_mgr = ctx->resource_manager();
    string container = container_.empty() ? r_mgr->default_container() : container_;
    string name = shared_name_.empty() ? strings::StrCat("_musa_table_", node_name_) : shared_name_;

    MusaTableResource<K, V>* table = nullptr;

    // 🌟 创建函数：直接创建我们的包装类
    auto create_fn = [](MusaTableResource<K, V>** t) {
      *t = new MusaTableResource<K, V>();
      return Status::OK();
    };

    OP_REQUIRES_OK(ctx, r_mgr->LookupOrCreate<MusaTableResource<K, V>>(
                            container, name, &table, create_fn));
    
    // LookupOrCreate 增加引用，必须释放
    core::ScopedUnref unref_me(table);

    Tensor* handle_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle_tensor));
    
    // 关键：创建一个 Handle 指向我们的 MusaTableResource
    handle_tensor->flat<ResourceHandle>()(0) = 
        MakeResourceHandle<MusaTableResource<K, V>>(ctx, container, name);
  }

 private:
  string container_;
  string shared_name_;
  string node_name_;
};

// 3. 必须重新实现 Insert 算子，因为原生的 InsertV2 不认识 MusaTableResource
template <typename K, typename V>
class MusaHashTableInsertOp : public MusaOpKernel {
 public:
  explicit MusaHashTableInsertOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    MusaTableResource<K, V>* table = nullptr;
    // 从输入的 handle 中找资源
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &table));
    core::ScopedUnref unref_me(table);

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);

    const auto key_flat = keys.flat<K>();
    const auto val_flat = values.flat<V>();

    mutex_lock l(table->mu);
    for (int i = 0; i < key_flat.size(); ++i) {
      table->data[key_flat(i)] = val_flat(i);
    }
  }
};

// ================= 注册所有相关算子 =================

#define REGISTER_MUSA_TABLE_OPS(K, V)                                \
  REGISTER_KERNEL_BUILDER(Name("MutableHashTableV2")                 \
                              .Device(DEVICE_MTGPU)                  \
                              .TypeConstraint<K>("key_dtype")         \
                              .TypeConstraint<V>("value_dtype")       \
                              .HostMemory("table_handle"),           \
                          MusaMutableHashTableOp<K, V>);             \
  REGISTER_KERNEL_BUILDER(Name("LookupTableInsertV2")                \
                              .Device(DEVICE_MTGPU)                  \
                              .TypeConstraint<K>("key_dtype")         \
                              .TypeConstraint<V>("value_dtype")       \
                              .HostMemory("table_handle")            \
                              .HostMemory("keys")                    \
                              .HostMemory("values"),                 \
                          MusaHashTableInsertOp<K, V>);

REGISTER_MUSA_TABLE_OPS(int64, float);
REGISTER_MUSA_TABLE_OPS(int32, float);

}  // namespace musa
}  // namespace tensorflow
