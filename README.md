# TensorFlow MUSA Extension (TF 2.15 variant)

面向摩尔线程（Moore Threads）MUSA GPU 的 TensorFlow 插件：通过 MUSA 内核与图优化为 TensorFlow 提供 GPU 加速。

> **这是 TF 2.15 版本。** 原始 2.6.1 版本仓库位于 `../tensorflow_musa_extension/`。
> 此目录是从 2.6.1 版本派生的脚手架移植 (scaffold port)，**尚未在 TF 2.15 上
> 完成构建与验证**。详见 [PORTING.md](PORTING.md) 了解已完成的变更以及构建
> 时预期会遇到的问题。同一份源代码理论上也可用于 TF 2.13 / 2.14，只需要把
> `setup.py` 和 `build.sh` 里的 `REQUIRED_TF_VERSION` 改为对应版本号即可。

## 特性

- 核心算子与常用融合路径的 MUSA 实现
- Grappler 图优化（布局、融合、可选混合精度等）
- Python 包 `tensorflow_musa`：自动加载插件与设备查询
- bf16 / 混合精度专用优化：`MusaResourceApplyAdamMixed` op、fp32 内部累加的
  bf16 Adam 路径、RNE bf16 取整、Sum/Mean/Prod 的 fp32 提升、AddV2/AddN/Mul
  bf16 向量化快路径、bf16↔fp32 向量化 Cast 等
- 可选遥测与调试说明见 [调试指南](docs/DEBUG_GUIDE.md)

## 环境要求

- CMake ≥ 3.10，Make，GCC/G++ 11+（与 TensorFlow 2.15.x wheel ABI 一致）
- MUSA SDK（默认路径 `/usr/local/musa`）：Runtime、muBLAS、muDNN
- Python ≥ 3.9（TF 2.15 不再支持 3.8）
- **TensorFlow == 2.15.1**（须与此版本一致）
- NumPy ≥ 1.23, < 2.0（TF 2.15 wheel 与 NumPy 2.x ABI 不兼容）

## 安装（推荐：Wheel）

```bash
git clone <repository-url>
cd tensorflow_musa_extension_2.15

pip install tensorflow==2.15.1
./build.sh wheel
pip install dist/tensorflow_musa-*.whl --no-deps
```

构建前请阅读 [PORTING.md](PORTING.md)，了解从 2.6.1 移植到 2.15 时已应用的
变更，以及第一次构建可能遇到的问题（主要在 `musa_ext/mu/device/` 下的
StreamExecutor 适配代码，因 TF 2.10-2.15 期间 PluggableDevice C++ API 有
若干增减；以及 `Status::OK()` / `Status::error_message()` 在 TF 2.14+ 被
deprecation 警告标记，但仍可正常编译运行）。

重新构建后覆盖安装可加 `--force-reinstall`。

## 快速验证

```python
import tensorflow_musa as tf_musa

print(tf_musa.__version__)
print(tf_musa.get_musa_devices())
```

在计算图中使用 MUSA 设备（示例）：

```python
import tensorflow as tf
import tensorflow_musa  # 确保插件已加载

with tf.device("/device:MUSA:0"):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.matmul(a, a)
```

### MUSA 显存按需增长

`tensorflow_musa` 支持控制 MUSA BFC allocator 的 `allow_growth` 行为。默认值与 TensorFlow 原生 GPU 保持一致，为 `False`；启用后，MUSA 显存池会按需增长，而不是在设备初始化时一次性申请完整显存池。请在 MUSA 设备初始化前设置：

```python
import tensorflow_musa as tf_musa

tf_musa.set_musa_allow_growth(enabled=True)
```

如需显式关闭：

```python
tf_musa.set_musa_allow_growth(enabled=False)
```

也可以使用 TensorFlow 官方兼容环境变量强制覆盖 Python 设置：

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### MUSA 自定义图优化器开关

`tensorflow_musa` 提供了 `ConfigProto` 级别的接口，用于启用、关闭或查询 `musa_graph_optimizer`。常规推理场景推荐使用 `enable_musa_graph_optimizer(config)`，它等价于向 `config.graph_options.rewrite_options.custom_optimizers` 注册 `musa_graph_optimizer`。

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

config = tf.compat.v1.ConfigProto()

# 启用 MUSA 自定义图优化器
tf_musa.enable_musa_graph_optimizer(config)

# 查询是否已启用
print(tf_musa.is_musa_graph_optimizer_enabled(config))

# 关闭 MUSA 自定义图优化器
tf_musa.disable_musa_graph_optimizer(config)
```

也可以使用统一接口显式传入开关值：

```python
tf_musa.set_musa_graph_optimizer_enabled(config, enabled=True)
tf_musa.set_musa_graph_optimizer_enabled(config, enabled=False)
```

按名称关闭部分融合 pattern 时，可直接在 Python 配置里传参给 C++ 优化器：

```python
tf_musa.disable_musa_fusion_patterns(
    config,
    patterns=["MusaGeluFusion", "MusaLayerNormFusion"],
)

# 关闭所有融合 pattern
tf_musa.disable_musa_fusion_patterns(config, patterns="all")

# 清除融合 pattern 禁用列表
tf_musa.clear_musa_disabled_fusion_patterns(config)
```

少数测试或调试场景需要强制设置 Grappler optimizer 列表时，可以额外传入 `add_to_optimizer_list=True`：

```python
tf_musa.enable_musa_graph_optimizer(config, add_to_optimizer_list=True)
```

## 从源码构建插件（可选）

仅生成 `build/libmusa_plugin.so`（不打包 wheel）：

```bash
pip install tensorflow==2.6.1
./build.sh          # 或 ./build.sh release
```

开发时也可在 Python 中 `tf.load_library("./build/libmusa_plugin.so")` 手动加载。

## 文档与示例

- [调试与环境变量](docs/DEBUG_GUIDE.md)
- 更多示例：[TensorFlow MUSA Playground](https://gitee.com/mthreadsacademy/tensorflow_musa_playground)

## 参与贡献

欢迎提交 Issue 与 Pull Request（新算子请附带测试）。

## 许可证

Apache License 2.0

## 支持

请在仓库 Issue 中反馈问题或联系维护者。
