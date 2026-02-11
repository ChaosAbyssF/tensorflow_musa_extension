/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include "utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/kernels/lookup_util.h" 
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace musa {

using LookupInterface = lookup::LookupInterface;

// ==================== 1. FindV2 (查找) ====================
template <typename K, typename V>
class MusaLookupTableFindOp : public OpKernel {
 public:
  explicit MusaLookupTableFindOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    LookupInterface* table = nullptr;
    OP_REQUIRES_OK(c, lookup::GetLookupTable("table_handle", c, &table));
    core::ScopedUnref unref_me(table);

    const Tensor& keys = c->input(1);
    const Tensor& default_value = c->input(2);

    TensorShape output_shape = keys.shape();
    output_shape.AppendShape(table->value_shape());

    Tensor* values = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &values));

    if (keys.NumElements() > 0) {
      OP_REQUIRES_OK(c, table->Find(c, keys, values, default_value));
    }
  }
};

// ==================== 2. InsertV2 (插入) ====================
template <typename K, typename V>
class MusaLookupTableInsertOp : public OpKernel {
 public:
  explicit MusaLookupTableInsertOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    LookupInterface* table = nullptr;
    OP_REQUIRES_OK(c, lookup::GetLookupTable("table_handle", c, &table));
    core::ScopedUnref unref_me(table);

    const Tensor& keys = c->input(1);
    const Tensor& values = c->input(2);

    OP_REQUIRES_OK(c, table->CheckKeyAndValueTensorsForInsert(keys, values));
    OP_REQUIRES_OK(c, table->Insert(c, keys, values));

    if (c->num_outputs() > 0) {
      c->set_output(0, c->input(0));
    }
  }
};

// ==================== 3. RemoveV2 (删除) ====================
template <typename K, typename V>
class MusaLookupTableRemoveOp : public OpKernel {
 public:
  explicit MusaLookupTableRemoveOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    LookupInterface* table = nullptr;
    OP_REQUIRES_OK(c, lookup::GetLookupTable("table_handle", c, &table));
    core::ScopedUnref unref_me(table);

    const Tensor& keys = c->input(1);
    OP_REQUIRES_OK(c, table->CheckKeyTensorForRemove(keys));
    OP_REQUIRES_OK(c, table->Remove(c, keys));

    if (c->num_outputs() > 0) {
      c->set_output(0, c->input(0));
    }
  }
};

// ==================== 4. ExportV2 (导出) ====================
class MusaLookupTableExportOp : public OpKernel {
 public:
  explicit MusaLookupTableExportOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    LookupInterface* table = nullptr;
    OP_REQUIRES_OK(c, lookup::GetLookupTable("table_handle", c, &table));
    core::ScopedUnref unref_me(table);

    // 【核心修正】：根据编译器 note，ExportValues 只接收一个 ctx 参数。
    // 在这个接口设计中，table->ExportValues(c) 内部会自动调用 
    // c->allocate_output 或 c->set_output 来填充输出的 keys 和 values。
    OP_REQUIRES_OK(c, table->ExportValues(c));
  }
};

// ==================== 注册区 ====================
#define REGISTER_MUSA_LOOKUP_OPS(K, V)                                         \
  REGISTER_KERNEL_BUILDER(Name("LookupTableFindV2")                            \
                              .Device(DEVICE_MTGPU)                            \
                              .HostMemory("table_handle")                      \
                              .TypeConstraint<K>("Tin")                        \
                              .TypeConstraint<V>("Tout"),                      \
                          MusaLookupTableFindOp<K, V>);                        \
  REGISTER_KERNEL_BUILDER(Name("LookupTableInsertV2")                          \
                              .Device(DEVICE_MTGPU)                            \
                              .HostMemory("table_handle")                      \
                              .TypeConstraint<K>("Tin")                        \
                              .TypeConstraint<V>("Tout"),                      \
                          MusaLookupTableInsertOp<K, V>);                      \
  REGISTER_KERNEL_BUILDER(Name("LookupTableRemoveV2")                          \
                              .Device(DEVICE_MTGPU)                            \
                              .HostMemory("table_handle")                      \
                              .TypeConstraint<K>("Tin"),                       \
                          MusaLookupTableRemoveOp<K, V>);

REGISTER_MUSA_LOOKUP_OPS(int32, float);
REGISTER_MUSA_LOOKUP_OPS(int64, float);
REGISTER_MUSA_LOOKUP_OPS(int32, int32);
REGISTER_MUSA_LOOKUP_OPS(int64, int64);
REGISTER_MUSA_LOOKUP_OPS(tstring, float);

REGISTER_KERNEL_BUILDER(Name("LookupTableExportV2")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("table_handle")
                            .HostMemory("keys")
                            .HostMemory("values"), 
                        MusaLookupTableExportOp);

} // namespace musa
} // namespace tensorflow
