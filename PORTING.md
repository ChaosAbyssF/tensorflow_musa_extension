# Porting the MUSA TF Plugin from TF 2.6.1 to TF 2.15

This directory is a **scaffold port** of the original plugin (sister directory
`tensorflow_musa_extension/`, which targets TF 2.6.1) to TF 2.15. The goal is
to make as few source changes as possible while moving the build to a TF
version that has substantially better native bf16 support in stock kernels.

The same source tree should also work for **TF 2.13** and **TF 2.14** with a
1-line change to `REQUIRED_TF_VERSION` in `setup.py` and `build.sh`, because
the ABI flag, plugin C++ API surface, and Keras `legacy.Adam` are stable
across that window. The scaffold targets 2.15.1 because it's the latest of
the three and gets the most upstream-fix backports.

> **Status: source-audited, not yet built.** Every mechanical change
> that the TF 2.15.1 source mandates has been applied, including the SE
> interface drift (cross-referenced against
> `third_party/xla/xla/stream_executor/stream_executor_internal.h` at
> the `v2.15.1` tag). The plugin has not yet been compiled against a TF
> 2.15 wheel nor run on MUSA hardware. The first build attempt may
> still surface 1-2 specific bf16 or muDNN-vendor issues, but the
> framework-API drift catalogued in §3 should all be resolved.

---

## 1. What's already done

| Change | File | Why |
|---|---|---|
| `_GLIBCXX_USE_CXX11_ABI` flipped from `=0` to `=1` | `CMakeLists.txt` | TF 2.10+ official wheels ship with the new C++11 ABI. Mixing ABIs across the plugin / framework boundary causes link-time `undefined reference to std::__cxx11::basic_string` errors. |
| `REQUIRED_TF_VERSION` bumped to `2.15.1` | `setup.py`, `build.sh` | Wheel build will refuse to proceed if TF isn't this version, catching environment misconfigurations early. Drop to `2.13.x` / `2.14.x` if you'd rather target those — no other source change required. |
| Package `__version__` bumped to `0.3.0` | `python/__init__.py` | Distinguishes the 2.15 variant from the 2.6.1 variant on PyPI / `pip list`. |
| `MusaAdam` base class resolved at runtime | `python/_optimizers.py` | TF 2.11 moved the old optimizer hierarchy under `tf.keras.optimizers.legacy.*`. The new `tf.keras.optimizers.Optimizer` base class has no `_resource_apply_dense` hook, so subclassing it the same way would silently break. The new `_resolve_adam_base_class()` helper picks `tf.keras.optimizers.legacy.Adam` when available, falls back to the plain class otherwise. |
| `Status::OK()` → `Status()` and `Status(error::Code, msg)` → `errors::*(msg)` migration | all `.cc` and `.mu` files under `musa_ext/` | TF 2.15 removed both the static `OK()` factory and the 2-argument `Status(error::Code, msg)` constructor from `absl::Status`. Affects 211 call sites across 64 files; resolved by a scripted perl sweep documented in §3.3. |
| `::uint4` namespace qualification | `musa_ext/kernels/math/musa_{add,mul,addn}_kernel.mu`, `musa_ext/kernels/math/musa_cast_kernels.mu` | TF 2.15 pulls in `ml_dtypes::uint4` (a 4-bit integer type) through `tensorflow/core/framework/types.h`, shadowing the CUDA/MUSA built-in 16-byte `uint4` vector type. All vec8 fast paths reference `::uint4` explicitly to force global-namespace lookup. See §3.1b. |
| `tensorflow/stream_executor/` → `xla/stream_executor/` mass rename | all files under `musa_ext/mu/device/` and `musa_ext/mu/device_register.cc` | TF 2.14+ moved the SE source under `third_party/xla/`; the wheel installs at `tensorflow/include/xla/stream_executor/`. Mechanical sed sweep applied. The deleted `lib/{status,statusor,error}.h` headers are replaced with explicit `<absl/status/status.h>` and `<absl/status/statusor.h>` includes in `musa_executor.h`, `musa_platform.cc`, and `musa_stream.h`. |
| `MusaExecutor` overrides matched to 2.15 `StreamExecutorInterface` | `musa_ext/mu/device/musa_executor.h` | Several pure-virtuals were removed (`AllocateTimer`/`StartTimer`/`StopTimer`/`PlatformDeviceCount`/`GetTimerImplementation`); `HostCallback` switched from `std::function<absl::Status()>` to `absl::AnyInvocable<absl::Status() &&>`; `Memset(uint8, …)` is new (defaulted). All resolved against the 2.15.1 source. See §3.1 for the audit table. |
| `GpuDeviceInfo` → `AcceleratorDeviceInfo` rename | `musa_ext/mu/device/musa_device.{h,cc}` | TF 2.13 generalized the per-device GPU info struct + accessor to support non-NVIDIA accelerators. 3 call sites: struct member, virtual override return type, and the setter on `DeviceBase`. See §3.2b. |
| `PluginConfig` removal | `musa_ext/mu/device/musa_executor.h`, `musa_ext/mu/device/musa_platform.cc` | TF 2.15 deleted `PluginConfig` from `StreamExecutorConfig` and removed `Platform::ExecutorForDeviceWithPluginConfig()` along with the trace-listener virtuals (`Register/UnregisterTraceListener`). Trace listeners moved to `StreamExecutorInterface`. We dropped the override + member entirely; `MusaExecutor` is now default-constructible. See §3.1c. |
| `allocate_temp` return value checked in `MusaFusedBatchNormOp` | `musa_ext/kernels/nn/musa_fused_batchnorm_op.cc` | `tsl::Status` became `[[nodiscard]]` in 2.15; two unwrapped calls were silently masking allocator OOM. Wrapped with `OP_REQUIRES_OK` (real correctness fix, not just a warning silencer). See §3.3. |
| `tensorflow/core/lib/core/errors.h` pulled into `fusion_pattern_manager.h` | `musa_ext/graph/fusion/fusion_pattern_manager.h` | The Status migration replaced `Status(error::INVALID_ARGUMENT, msg)` with `errors::InvalidArgument(msg)` across the 17 fusion `.cc` files. Those files never included an `errors.h` (they got the legacy 2-arg `Status` constructor transitively via `tensorflow/core/lib/core/status.h`). Adding the include to the shared `fusion_pattern_manager.h` resolves all 17 in one shot. See §3.3b. |
| `StatusOr::ValueOrDie()` → `*statusor` (operator deref) | `musa_ext/mu/device_register.cc` | `absl::StatusOr` (= `tsl::StatusOr`) removed `ValueOrDie()` in TF 2.13+. Both sites already check `.ok()` first, so the change is purely syntactic. See §3.1d. |
| `Stream::Stream(executor, impl)` → pending-handle hand-off through `MusaExecutor` | `musa_ext/mu/device/musa_executor.h`, `musa_ext/mu/device/musa_device.cc` | TF 2.15 changed the Stream constructor to take only `StreamExecutor*` and to internally call `executor->implementation()->GetStreamImplementation()` to build/own its `StreamInterface`. To keep the compute stream consistent between the SE Stream object and the rest of `MusaDevice` (muDNN / muBLAS / direct musa API calls), `MusaDeviceContext` now stages its handle on the executor via `MusaExecutor::SetPendingStreamHandle()` just before constructing the SE Stream; the next `GetStreamImplementation()` call consumes the staged handle instead of creating a fresh `musaStream_t`. Single-threaded setup hand-off only — see §3.1e. |
| `BFCAllocator(SubAllocator*, size_t, bool, name, bool)` → `BFCAllocator(unique_ptr<SubAllocator>, size_t, name, Options)` | `musa_ext/mu/device/musa_device.cc` | TF 2.15 changed the BFCAllocator constructor: SubAllocator is now passed as `unique_ptr<>`, and `allow_growth` + `garbage_collection` moved into a packed `BFCAllocator::Options` struct. Two call sites (compute + host pinned) updated. See §3.1f. |

These cover **every API drift point that the 2.15.1 source mandates**.
The list was assembled by walking the headers
`third_party/xla/xla/stream_executor/stream_executor_internal.h` and
`tensorflow/include/tsl/platform/status.h` in the official tarball and
diffing against the 2.6.1 plugin's overridden surface. There is no
known further mechanical drift to chase on TF 2.15.1 specifically; any
remaining build issues are environmental (NumPy ABI, muDNN ABI, MUSA
toolkit version).

---

## 2. What to do next

### Step 1 — install TF 2.15 in a fresh venv

```bash
python -m venv .venv-2.15
source .venv-2.15/bin/activate
pip install --upgrade pip setuptools wheel
# Pin NumPy 1.x BEFORE installing TF — TF 2.15 wheels are NumPy 1.x-ABI
# and refuse to import on NumPy 2.x with a clear error message.
pip install "numpy>=1.23,<2.0"
pip install tensorflow==2.15.1
```

Note that TF 2.15 supports Python 3.9 - 3.11. If your current env uses
Python 3.7 (the 2.6.1 default) or 3.8 (the 2.13 floor), you'll need a
newer Python interpreter.

### Step 2 — first build attempt

```bash
cd /root/code-space/tensorflow_musa_extension_2.15
./build.sh 2>&1 | tee build.log
```

Expect failures. Don't panic. Most fall into the categories below.

### Step 3 — work through the expected-failure categories

The following are **anticipated** failure modes. Each has a recipe. None
are research projects — they're all small concrete edits.

---

## 3. Expected failure categories (and how to resolve)

### 3.1 StreamExecutor / Platform API drift (PATCHED IN THIS SCAFFOLD)

**Files:** `musa_ext/mu/device/musa_platform.cc`,
`musa_ext/mu/device/musa_executor.h`, `musa_ext/mu/device/musa_stream.h`,
and others under `mu/device/`.

The plugin uses the **internal C++ StreamExecutor** classes directly:

```cpp
class MusaPlatform : public se::Platform { ... };
class MusaExecutor : public se::internal::StreamExecutorInterface { ... };
class MusaStream   : public se::internal::StreamInterface { ... };
```

Between TF 2.6 and TF 2.15, that interface was touched many times as part
of the PluggableDevice / PJRT migration. The following items are the
**actual** drifts that the 2.15.1 source mandates (audited against
`third_party/xla/xla/stream_executor/stream_executor_internal.h` in the
released `v2.15.1` tag). All have been resolved in this scaffold.

| Change | What's different in 2.15 | Fix applied |
|---|---|---|
| `internal::TimerInterface` removed | The whole timing-via-Timer machinery is gone; SE now measures via Events. | Removed `GetTimerImplementation`, `AllocateTimer`, `DeallocateTimer`, `StartTimer`, `StopTimer` overrides from `MusaExecutor`. |
| `PlatformDeviceCount` removed | Counting is done by `VisibleDeviceCount()` on `Platform` only. | Removed override from `MusaExecutor`. |
| `HostCallback` signature | Was `bool HostCallback(Stream*, std::function<absl::Status()>)`. Now `bool HostCallback(Stream*, absl::AnyInvocable<absl::Status() &&>)`. The new type is move-only and rvalue-call-only — you have to `std::move(*cb)()` to invoke it. | Override rewritten to match; explicit `<absl/functional/any_invocable.h>` include added. |
| `Memset(uint8 pattern, …)` newly present (defaulted) | New optional override; default impl returns `"Not implemented"`. | Wired through to `tensorflow::musa::Memset` for parity with `MemZero` so XLA/SE-internal callers don't get an error path. |
| `Allocate(uint64, int64)` → `Allocate(uint64_t, int64_t)` | Upstream uses `_t`-suffixed types. `xla/stream_executor/platform/port.h` re-exports `uint64`/`uint32`/`uint8` into `namespace stream_executor`, **but not `int64`**. | Plain `int64` (and `uint64` for consistency in this file) changed to `_t` suffixed forms throughout `MusaExecutor`. |
| Header `tensorflow/stream_executor/lib/{status,statusor,error}.h` deleted | The whole `stream_executor/lib/` subdir is gone in 2.15. `port::Status` / `port::StatusOr` therefore have no provider. | Migrated all SE adapters to `absl::Status` / `absl::StatusOr`; added explicit `<absl/status/status.h>` and `<absl/status/statusor.h>` to `musa_executor.h`, `musa_platform.cc`, and `musa_stream.h`. |
| Header `tensorflow/stream_executor/*.h` moved to `xla/stream_executor/*.h` | Whole namespace `stream_executor` source now lives at `third_party/xla/xla/stream_executor/` and is installed at `tensorflow/include/xla/stream_executor/` in the wheel. | Mechanical sed sweep over `musa_ext/mu/device/` and `musa_ext/mu/device_register.cc`: `tensorflow/stream_executor/` → `xla/stream_executor/`. |
| `xla/stream_executor/platform/port.h` thinned out | The file still exists but only re-exports `tsl::int8 … uint64`. The `port::Status` / `port::StatusOr` aliases and `SE_ASSIGN_OR_RETURN` macro are gone. | Not directly included anywhere in the plugin; types come transitively through `stream_executor_internal.h`. |

The `MusaStream` interface itself (`internal::StreamInterface`) is much
simpler in 2.15: only `GpuStreamHack()` and `GpuStreamMemberHack()` are
optional overrides, both already provided. There are no new pure
virtuals to satisfy.

The `MusaPlatform` interface (`se::Platform`) is largely unchanged
between 2.6 and 2.15; the existing overrides for `VisibleDeviceCount`,
`Id`, `Name`, `GetExecutor`, `GetUncachedExecutor`, and the trace
listener stubs continue to compile clean.

**Recipe (for future drift):** for any new compile error in this group,
open the corresponding TF 2.15 header — the one source of truth for SE
internals is
`<TF_SOURCE>/third_party/xla/xla/stream_executor/stream_executor_internal.h`
or, on an installed wheel,
`<venv>/lib/python*/site-packages/tensorflow/include/xla/stream_executor/stream_executor_internal.h`
— and match the override to the current virtual declaration.

If a future TF release moves more pieces around (the path drift from
`tensorflow/stream_executor/` → `xla/stream_executor/` happened mid-2.x,
the `lib/` subdir deletion shortly after), the alternative escape hatch
is to migrate to the **C-API PluggableDevice** model. That's a bigger
refactor (separate `.so` for device + ops; registration via
`SE_RegisterPlatform`), but it's the long-term supported path and is
forward-compatible with any TF 2.10+ version. For 2.15 specifically, the
internal C++ path still works, so try this scaffold first.

### 3.1b `ml_dtypes::uint4` vs CUDA `uint4` collision (already patched)

**Files:** `musa_ext/kernels/math/musa_add_kernel.mu`,
`musa_ext/kernels/math/musa_mul_kernel.mu`,
`musa_ext/kernels/math/musa_cast_kernels.mu`,
`musa_ext/kernels/math/musa_addn_kernel.mu`.

TF 2.15 pulls in the `ml_dtypes` library (FP8 / INT4 / etc. low-precision
ML types). It defines `ml_dtypes::uint4` as a **4-bit integer type** and
brings it into scope via a `using` declaration transitively included by
`tensorflow/core/framework/types.h`. This shadows the CUDA/MUSA built-in
`uint4` (a 16-byte vector struct with `.x/.y/.z/.w` of `unsigned int`).

The `.mu` files listed above use `uint4` as a 16-byte vectorized load type
in the vec8 fast paths. Without qualification, the compiler picks
`ml_dtypes::uint4` and errors with:

```
error: no member named 'x' in 'ml_dtypes::i4<unsigned char>'
    out.x = add_bf16_pair_packed(l.x, r.x);
```

**Resolution (already applied in this scaffold):** every `uint4` use in the
new kernels is qualified as `::uint4` to force global-namespace lookup of
the CUDA built-in. This matches the convention the existing `musa_neg_kernel.mu`
already used. If you add a new `.mu` file that does 16-byte vector loads,
write `::uint4` from day one.

If you also encounter the same pattern with `int4` (ml_dtypes also defines
that as a 4-bit signed type), the fix is the same: qualify as `::int4`.

**Files:** any `.cc` or `.mu` under `musa_ext/kernels/` and `musa_ext/graph/`.

Audited against the released `v2.15.1` source tree, **every**
`tensorflow/core/*` and `tensorflow/c/*` header we depend on still exists
at its old path:

```bash
# Verification command — should print "0 missing"
TFSRC=/path/to/tensorflow-2.15.1
HEADERS=$(grep -rhoE 'tensorflow/[a-z_/]+\.h' --include='*.cc' \
            --include='*.h' --include='*.mu' musa_ext/ | sort -u)
missing=0; for h in $HEADERS; do
  [ ! -f "$TFSRC/$h" ] && { echo "MISSING $h"; missing=$((missing+1)); }
done; echo "$missing missing"
```

Verified-stable in 2.15:

- `tensorflow/core/framework/{op_kernel,tensor,types,resource_var,…}.h`
- `tensorflow/core/lib/core/{errors,status,notification,threadpool}.h`
- `tensorflow/core/lib/random/{philox_random,random_distributions}.h`
- `tensorflow/core/platform/{env,errors,logging,mutex,status,types,stream_executor}.h`
- `tensorflow/core/grappler/optimizers/custom_graph_optimizer{,_registry}.h`
- `tensorflow/core/kernels/{gpu_prim,tensor_list}.h`
- `tensorflow/c/{kernels,experimental/stream_executor/stream_executor}.h`

The only headers that **did** move/disappear belong to StreamExecutor
internals and are documented in section 3.1:

- `tensorflow/stream_executor/*.h` → `xla/stream_executor/*.h`
  (used by `musa_ext/mu/device/`; mechanical sed sweep already applied).
- `tensorflow/stream_executor/lib/{status,statusor,error}.h` → **deleted**.
  Replaced with `<absl/status/status.h>` / `<absl/status/statusor.h>`.
- `tensorflow/stream_executor/platform/port.h` → moved to
  `xla/stream_executor/platform/port.h` and stripped: it only re-exports
  integer aliases (`tsl::int8 … tsl::uint64`) into the `stream_executor`
  namespace. `port::Status` and `port::StatusOr` are gone — we don't use
  them anywhere any more.

**Recipe (only if more drift surfaces in a future patch release):** if a
specific include doesn't resolve, search the wheel's include tree:

```bash
TF_INC=$(python -c "import tensorflow as tf; print(tf.sysconfig.get_include())")
find "$TF_INC" -name $(basename <missing_header>) -print
```

and update the path. The first build's output will list exactly which
includes broke.

### 3.1c `PluginConfig` removed from StreamExecutor surface (PATCHED)

**Files:** `musa_ext/mu/device/musa_executor.h`,
`musa_ext/mu/device/musa_platform.cc`.

TF 2.15 deleted three related pieces of the SE plugin-config machinery
as part of the C-API PluggableDevice migration:

1. **`PluginConfig` struct itself** — gone from `StreamExecutorConfig`.
   The struct now holds only `gpu_stream`, `ordinal`, and
   `device_options`.
2. **`Platform::ExecutorForDeviceWithPluginConfig(int, const PluginConfig&)`** —
   removed.  Callers should use `ExecutorForDevice(ordinal)` only.
3. **`Platform::RegisterTraceListener()` / `UnregisterTraceListener()`** —
   removed from `Platform`.  Trace listeners now register on the
   `StreamExecutorInterface` directly.

Build errors before the fix:

```
error: 'PluginConfig' does not name a type
error: '…ExecutorForDeviceWithPluginConfig' marked 'override', but does not override
error: '…RegisterTraceListener' marked 'override', but does not override
error: 'struct StreamExecutorConfig' has no member named 'plugin_config'
```

**Resolution applied:**
- `MusaExecutor` constructor now takes no arguments (was
  `explicit MusaExecutor(const PluginConfig&)`).  The `plugin_config_`
  data member was removed; `device_ordinal_` is the only remaining
  state and is set inside `Init()`.
- `MusaPlatform::ExecutorForDeviceWithPluginConfig()` deleted.
- `MusaPlatform::RegisterTraceListener()` / `UnregisterTraceListener()`
  deleted.
- `MusaPlatform::GetUncachedExecutor()` builds the executor with
  `std::make_unique<MusaExecutor>()` (no args) instead of passing
  `config.plugin_config`.

The cascading compile failures the build reported across
`device_register.cc.o`, `musa_device.cc.o`, `musa_random_op.cc.o`, etc.
all stemmed from the same header (`musa_executor.h`) being included
transitively — once that header compiles, the cascade clears.

### 3.1d `StatusOr::ValueOrDie()` removed (PATCHED)

**Files:** `musa_ext/mu/device_register.cc`.

`absl::StatusOr` deprecated `ValueOrDie()` in mid-2021 and TF 2.13+ removed
it from the public surface (`tsl::StatusOr` aliases to `absl::StatusOr`).
Replacement: `*statusor` (operator dereference) or `statusor.value()` —
both abort on a `!ok()` payload, matching the historical `ValueOrDie()`
semantics.  Two call sites in `MusaDeviceFactory::CreateDevices` switched
to `*` deref; both already check `.ok()` immediately before, so the change
is purely syntactic.

Build error before the fix:

```
error: 'class absl::lts_20230125::StatusOr<stream_executor::Platform*>'
       has no member named 'ValueOrDie'
   95 |     auto* platform = platform_status.ValueOrDie();
```

### 3.1e `Stream` constructor signature change (PATCHED)

**Files:** `musa_ext/mu/device/musa_executor.h`,
`musa_ext/mu/device/musa_device.cc`.

| TF 2.6 | TF 2.15 |
|---|---|
| `Stream(StreamExecutor* parent, StreamInterface* impl)` | `Stream(StreamExecutor* parent)` |
| Stream borrows your impl pointer | Stream **owns** an impl it builds via `parent->implementation()->GetStreamImplementation()` |

This change closed off the path of injecting a pre-existing
`musaStream_t` into the SE Stream object: the Stream insists on creating
its own impl through the executor.  If we let it do that naively, the
resulting `official_stream_` would wrap a fresh `musaStream_t` while
`MusaDevice::stream_` (already bound to muDNN / muBLAS) would point at a
different one — kernel work submitted via the SE Stream would race with
muDNN / muBLAS operations.

**Resolution applied:** a single-shot "pending handle" hand-off through
`MusaExecutor`:

```cpp
class MusaExecutor : public internal::StreamExecutorInterface {
 public:
  void SetPendingStreamHandle(musaStream_t h) { pending_stream_handle_ = h; }
  std::unique_ptr<internal::StreamInterface>
  GetStreamImplementation() override {
    if (pending_stream_handle_) {
      musaStream_t h = pending_stream_handle_;
      pending_stream_handle_ = nullptr;
      return std::make_unique<MusaStream>(h);
    }
    musaStream_t h; musaStreamCreate(&h);
    return std::make_unique<MusaStream>(h);
  }
 private:
  musaStream_t pending_stream_handle_ = nullptr;
};
```

`MusaDeviceContext::MusaDeviceContext()` calls `SetPendingStreamHandle()`
on the executor immediately before `new Stream(executor)`, so the Stream's
internal `GetStreamImplementation()` call adopts the caller's handle
verbatim.  This is a single-threaded setup hand-off, not a general
thread-safe stream reuse mechanism — but the only caller is the
synchronous MusaDevice constructor.

Build error before the fix:

```
error: no matching function for call to 'stream_executor::Stream::Stream(
       stream_executor::StreamExecutor*&,
       stream_executor::internal::StreamInterface*&)'
```

### 3.1f `BFCAllocator` constructor signature change (PATCHED)

**Files:** `musa_ext/mu/device/musa_device.cc`.

| TF 2.6 | TF 2.15 |
|---|---|
| `BFCAllocator(SubAllocator*, size_t total, bool allow_growth, const string& name, bool garbage_collection)` | `BFCAllocator(unique_ptr<SubAllocator>, size_t total, const string& name, const Options& opts)` |

The `Options` struct packs `allow_growth`, `garbage_collection`,
`allow_retry_on_failure`, and `fragmentation_fraction`.  Two call sites
in `MusaDevice::MusaDevice` (the device allocator and the host-pinned
allocator) updated to the new signature with explicit `Options`
construction.

Build error before the fix:

```
error: no matching function for call to 'tsl::BFCAllocator::BFCAllocator(
       tensorflow::musa::MusaSubAllocator*, size_t&, bool&,
       const char [19], bool)'
```

### 3.2b `GpuDeviceInfo` → `AcceleratorDeviceInfo` rename (PATCHED)

**Files:** `musa_ext/mu/device/musa_device.h`, `musa_ext/mu/device/musa_device.cc`.

TF 2.13 renamed the per-device "GPU info" struct exposed by
`DeviceBase` as part of generalizing the path to non-NVIDIA
accelerators (TPU, ROCm, MUSA, …):

| TF 2.6 / 2.12 | TF 2.13+ |
|---|---|
| `struct GpuDeviceInfo` | `struct AcceleratorDeviceInfo` |
| `tensorflow_gpu_device_info()` virtual | `tensorflow_accelerator_device_info()` virtual |
| `set_tensorflow_gpu_device_info(...)` | `set_tensorflow_accelerator_device_info(...)` |

The struct also gained two PJRT-related fields (`pjrt_context`,
`use_pjrt_tensor_buffer`) that default to `nullptr` / `false`, so the
existing MUSA initialization (which only sets `stream`, `default_context`,
and `gpu_id`) is forward-compatible without changes.

Build error before the fix:

```
error: 'GpuDeviceInfo' does not name a type
   89 |   const GpuDeviceInfo* tensorflow_gpu_device_info() const override {
```

**Resolution applied:** mechanical rename of all three call sites in
`musa_device.{h,cc}` — no behavioral change.

### 3.3b `errors::*` not visible in grappler fusion files (PATCHED)

**Files:** `musa_ext/graph/fusion/fusion_pattern_manager.h` (single-point
fix).

The Status-API migration in §3.3 replaced
`Status(error::INVALID_ARGUMENT, msg)` (the 2-argument constructor TF 2.15
removed) with the modern `errors::InvalidArgument(msg)` factory.  The
17 fusion `.cc` files in `musa_ext/graph/fusion/` never explicitly
included an `errors.h` — in TF 2.6.1 they got the 2-arg `Status`
constructor transitively via `tensorflow/core/lib/core/status.h`, and
didn't need the `errors::*` factories at all.

After the migration, every `errors::InvalidArgument` /
`errors::AlreadyExists` call in those files emitted:

```
error: 'InvalidArgument' is not a member of 'tensorflow::errors'
```

There were ~50 such errors across the 17 files.

**Resolution applied:** add `#include "tensorflow/core/lib/core/errors.h"`
to `fusion_pattern_manager.h`.  All 17 fusion `.cc` files transitively
include that header via their per-fusion `.h`, so this single edit
resolves the whole category.  No per-file changes needed.

Kernel `.cc` files in `musa_ext/kernels/` were already covered because
`tensorflow/core/framework/op_kernel.h` transitively pulls
`tensorflow/core/lib/core/errors.h`.

### 3.3 OpKernel API drift (LOW-RISK)

**Files:** every `.cc` under `musa_ext/kernels/`.

The public `OpKernel`, `OpKernelContext`, `Tensor`, `REGISTER_OP`,
`REGISTER_KERNEL_BUILDER` API is **stable between 2.6 and 2.15**. The
plugin's ~80 kernel files should compile largely unchanged.

Known minor drift in 2.15 specifically:

- **Status API migration (already applied in this scaffold).** TF 2.14
  deprecated and TF 2.15 *removed* the `tensorflow::Status::OK()` static
  factory and the `tensorflow::Status(error::Code, const char*)` 2-arg
  constructor. The plugin was using both heavily (211 OK calls across 64
  files; 9 files with the 2-arg constructor). A perl-based sweep migrated
  them in-place to the modern equivalents:

  | Old | New | Notes |
  |---|---|---|
  | `Status::OK()` | `Status()` | `absl::Status` default-constructed is OkStatus; the namespace context (`tensorflow::`, `port::`, etc.) doesn't change semantics. |
  | `Status(error::INVALID_ARGUMENT, msg)` | `errors::InvalidArgument(msg)` | `tensorflow::errors::*` factories are present in all `namespace tensorflow {}` files via the existing `errors.h` include. |
  | `Status(error::ALREADY_EXISTS, msg)` | `errors::AlreadyExists(msg)` | |
  | `Status(error::INTERNAL, msg)` | `errors::Internal(msg)` | |
  | `Status(error::NOT_FOUND, msg)` | `errors::NotFound(msg)` | |
  | `port::Status(port::error::CODE, msg)` | `absl::CodeError(msg)` | Used by StreamExecutor adapters in `mu/device/`. `absl::*Error` is always visible because TF 2.15 typedefs `port::Status` → `tsl::Status` → `absl::Status`. |

  `error_message()` is *not* used by the plugin (verified by grep), so no
  cleanup needed there. If you add new code post-port, prefer the modern
  forms from day one — the legacy form is rejected by the compiler in 2.15
  with two specific errors:

  ```
  error: 'OK' is not a member of 'tsl::Status' {aka 'absl::Status'}
  error: no matching function for call to 'Status::Status(error::Code, const char[...])'
  ```

- `ResourceHandle::Init` / `LookupResource` signatures unchanged.
- `forward_input_or_allocate_output`: unchanged.
- `allocate_output`, `allocate_temp`: signature unchanged, but
  **`tsl::Status` is now `[[nodiscard]]`**, so any call site that
  discarded the return value before now emits a warning:

  ```
  warning: ignoring returned value of type 'tsl::Status', declared with
           attribute 'nodiscard' [-Wunused-result]
       ctx->allocate_temp(DT_FLOAT, scale.shape(), &temp_acc_mean);
  ```

  These are real correctness bugs (silently proceeds with an
  uninitialized `Tensor` on allocator OOM, leading to use-after-stale-
  pointer or downstream segfault). The known instances at the time of
  the port are:
  - `musa_ext/kernels/nn/musa_fused_batchnorm_op.cc:87-88` — **fixed**
    in place by wrapping both `allocate_temp` calls with
    `OP_REQUIRES_OK(ctx, …)`.
  - Several call sites under `musa_ext/kernels/training/musa_applyadam_op.cc`
    and `musa_applygradientdescent_op.cc` discard the return inside
    lambdas (e.g. `fill_scalar`). These are pre-existing and not yet
    fixed in this scaffold — the build proceeds with warnings because
    TF plugin builds don't enable `-Werror=unused-result`. If you do
    enable that flag later, wrap each discarded call with
    `OP_REQUIRES_OK(ctx, …)` (or have the lambda return a `Status`).
- Eigen `Tensor` API: unchanged.

### 3.4 Grappler / custom graph optimizer (LOW-RISK)

**Files:** `musa_ext/mu/optimizer/musa_graph_optimizer.cc`,
`musa_ext/graph/fusion/*.cc`.

`REGISTER_GRAPH_OPTIMIZER_AS` and `CustomGraphOptimizer` are stable in TF
2.15. The `Optimize(Cluster*, const GrapplerItem&, GraphDef*)` signature
is unchanged. The fusion pattern manager is internal to the plugin and
doesn't touch TF API surface.

The plugin's `MusaGraphOptimizer::Init(...)` takes a
`RewriterConfig_CustomGraphOptimizer*` — stable.

In TF 2.15 the same Grappler pass receives `Status` instead of
`tensorflow::Status` in some new helper signatures, but since they're
typedef-aliased that's transparent.

### 3.5 muDNN ABI (UNKNOWN-RISK)

**Not a TF issue, a vendor issue.**

The plugin links against the MUSA SDK's `libmudnn` / `libmudnncxx`. If the
SDK was built specifically against TF 2.6.1's old-ABI libstdc++, it may
not link against the new-ABI plugin built for TF 2.15. Symptoms: linker
errors mentioning `std::__cxx11::basic_string` from muDNN symbols.

**Recipe:** if you hit this, you need a MUSA SDK build that targets the
new ABI. Coordinate with the SDK provider.

If `mudnncxx` was built old-ABI but the plain `mudnn` was built with
extern-C interfaces only (no `std::string` across the boundary), CMake
will prefer `mudnncxx`. Force it to use `mudnn` instead by replacing this
block in `CMakeLists.txt`:

```cmake
if(MUDNNCXX_LIBRARY)
    set(MUSA_DNN_LIBRARY "${MUDNNCXX_LIBRARY}")
elseif(MUDNN_LIBRARY)
    ...
```

with `set(MUSA_DNN_LIBRARY "${MUDNN_LIBRARY}")` directly.

### 3.6 Python wheel ABI tags (LOW-RISK)

`setup.py` lets `bdist_wheel` infer the tag automatically. For TF 2.15 on
Linux x86_64 the typical tag is `cp310-cp310-manylinux2014_x86_64`. If pip
refuses to install with `incompatible platform tag`, force the right tag
in `setup.py` via:

```python
class BdistWheelCommand(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False  # ensure non-pure tag
```

Already present in the inherited setup. Should not need touching.

---

## 4. bf16 quality improvements: do they carry over?

**Yes.** Every fix in the 2.6.1 plugin variant is dtype-correctness work,
not TF-version specific. The carried-forward improvements include:

- `MusaResourceApplyAdamMixed` op + kernel and `MusaAdam` Python wrapper.
- `MusaResourceApplyAdam<bfloat16>` fp32-internal dispatch.
- RNE rounding fix across 11 custom `.mu` files.
- `RunReduceWithFP32Promotion` helper wired into Sum/Mean/Prod/Min/Max.
- AddV2 / AddN / Mul bf16+fp16 vectorized fast paths.
- bf16↔fp32 and fp16↔fp32 vectorized Cast kernels.
- Adam mixed-precision Grappler rewrite (`MusaAdamMixedFusion`).

These all live in `.cc` / `.mu` files that should compile unchanged in 2.15.

The one piece that needed an explicit 2.11+ accommodation is the Python
`MusaAdam` class, which is already addressed via `_resolve_adam_base_class`.

---

## 5. What stops being needed in 2.15

TF 2.15 has noticeably better stock bf16 support than 2.6.1, particularly:

- Stock CPU / CUDA kernels for `Sum<bfloat16>`, `Mean<bfloat16>`, etc.
  internally accumulate in fp32. (The MUSA kernels still go through this
  plugin, so our Item-#3 fix is still relevant for MUSA. But on CPU it
  becomes redundant.)
- `Cast` in 2.15 has explicit, vectorized bf16↔fp32 paths in the runtime.
- Keras mixed_bfloat16 policy is more polished and avoids a few of the
  spurious Cast ops 2.6.1 emitted around `BiasAdd` and `LayerNorm`.
- The `Cast + ResourceApplyAdam` pattern that our `MusaAdamMixedFusion`
  rewrites is still what Keras emits, so the fusion remains useful.

In short, **nothing you fixed in the 2.6.1 plugin becomes redundant on MUSA
in 2.15** — the redundant fixes are on the CPU/CUDA side, which we don't
provide kernels for. The MUSA device sees this plugin's kernels, so all
the bf16 work continues to earn its keep.

---

## 6. Diff-with-2.6.1 cheat sheet

To see what's different between this directory and the 2.6.1 plugin:

```bash
diff -ruN ../tensorflow_musa_extension/ ./ \
    --exclude='.git' --exclude='build' --exclude='dist' \
    --exclude='__pycache__' --exclude='PORTING.md' \
    --exclude='.venv-*' --exclude='*.log' | head -200
```

Today this prints the 5 mechanical changes listed in section 1 above. As
you work through the build, each fix-and-rebuild cycle will grow this diff.

---

## 7. When to consider abandoning the C++ plugin path

If section 3.1 (StreamExecutor drift) turns into more than ~2 days of
work, that's a signal that TF 2.10-2.15's churn made the internal-C++
plugin path unergonomic for MUSA. The two reasonable forks at that point:

1. **Stay on TF 2.6.1.** The bf16 issues are now fixed in the existing
   plugin variant; the only thing you gain from 2.15 is better stock CPU
   kernels (irrelevant for MUSA) and a newer Keras (worth swapping
   `legacy.Adam` for if you care about new optimizer features).

2. **Migrate to the C-API PluggableDevice model.** This is a ~3-week
   project that produces a forward-compatible plugin for any TF 2.10+
   version. The interface is documented at
   [TF_NewKernelBuilder / TF_OpDefinitionBuilder](https://www.tensorflow.org/install/source#tested_build_configurations).
   With this path you build the plugin once and any future TF release
   (2.15 → 2.16 → 2.17+ → 3.x) loads it as long as the C-ABI hasn't
   broken, which is the explicit guarantee of the modular plugin API.

I'd recommend evaluating after the first compile attempt: if you get
through the kernel `.cc` files and only StreamExecutor remains broken,
finish it (~2-3 days). If you also hit ~10+ kernel API breaks, option 2
starts to look better, especially since you'd be locked into TF 2.15
otherwise — the same drift will hit you again on the next bump.

---

## 8. TF 2.13 / 2.14 compatibility

The source tree as-is is compatible with TF 2.13 and 2.14 too, with one
edit each:

- `setup.py`: change `REQUIRED_TF_VERSION = "2.15.1"` to `"2.13.x"` /
  `"2.14.x"`.
- `build.sh`: same string in two places.

The ABI flag, plugin C++ API surface, and Keras `legacy.Adam` are
all stable across that 2.13-2.15 window. The reason this scaffold
explicitly targets 2.15.1 is that it's the latest of the three and
ships with the most upstream-fix backports.

For TF 2.16+ this approach **breaks**: the `Status::OK()` /
`error_message()` deprecations were removed, and `tf.keras.optimizers.legacy`
was deleted. Plan to either do the mechanical Status cleanup and migrate
to the new Keras optimizer API at that point, or move to the C-API
PluggableDevice path (section 7, option 2).
