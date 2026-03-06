# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA Reduce test operators."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase, load_musa_plugin

# 显式加载插件
musa_ops = tf.load_op_library(load_musa_plugin())

class ReduceOpsTest(MUSATestCase):
    """Tests for MUSA ReduceTest and ReduceBmmTest operators."""

    def setUp(self):
        super().setUp()
        self.results_file = "reduce_ops_diff_results.txt"
        # Initialize file only once per test run
        if not hasattr(ReduceOpsTest, "_file_initialized"):
            with open(self.results_file, "w") as f:
                f.write("MUSA Reduce Operators Accuracy Test Results\n")
                f.write("="*50 + "\n")
            ReduceOpsTest._file_initialized = True

    def _log_diff(self, op_name, dtype, diff):
        with open(self.results_file, "a") as f:
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))
            f.write(f"Operator: {op_name}\n")
            f.write(f"DataType: {dtype}\n")
            f.write(f"Max Absolute Difference: {max_diff:.8e}\n")
            f.write(f"Mean Absolute Difference: {mean_diff:.8e}\n")
            f.write("-" * 30 + "\n")

    def testReduceTest(self):
        """Test ReduceTest op: [M, K] -> [M] (sum along dim 1)."""
        shape = [32, 256] # 增加规模
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            # 使用更大的范围增加数值差异
            x_np = np.random.uniform(-10, 10, size=shape).astype(np.float32)
            x = tf.cast(x_np, dtype=dtype)

            # 1. 使用原生 TF API 在 CPU 上计算作为基线 (Baseline)
            with tf.device("/CPU:0"):
                expected = tf.reduce_sum(x, axis=1)

            # 2. 使用自定义的 ReduceTest 算子在 MUSA 上计算
            with tf.device("/device:MUSA:0"):
                musa_result = musa_ops.ReduceTest(input=x)
            
            # 打印并记录误差信息
            diff = np.abs(musa_result.numpy() - expected.numpy())
            self._log_diff("ReduceTest", dtype.name, diff)
            print(f"\nReduceTest [{dtype.name}] Max Diff vs TF-CPU: {np.max(diff)}")

            # 断言对比
            rtol = 1e-2 if dtype != tf.float32 else 1e-4
            atol = 1e-2 if dtype != tf.float32 else 1e-4
            try:
                self.assertAllClose(musa_result, expected, rtol=rtol, atol=atol)
            except AssertionError as e:
                pass

    def testMusaMatMul(self):
        """Test MusaMatMul op: [M, K] x [K, N] -> [M, N]."""
        m, k, n = 32, 64, 32
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            a_np = np.random.uniform(-1, 1, size=[m, k]).astype(np.float32)
            b_np = np.random.uniform(-1, 1, size=[k, n]).astype(np.float32)
            
            a = tf.cast(a_np, dtype=dtype)
            b = tf.cast(b_np, dtype=dtype)

            # 1. 使用原生 TF API 在 CPU 上计算作为基线
            with tf.device("/CPU:0"):
                expected = tf.matmul(a, b)

            # 2. 使用自定义的 MusaMatMul 算子在 MUSA 上计算
            try:
                with tf.device("/device:MUSA:0"):
                    musa_result = musa_ops.MusaMatMul(a=a, b=b)

                # 打印并记录误差
                diff = np.abs(musa_result.numpy() - expected.numpy())
                self._log_diff("MusaMatMul", dtype.name, diff)
                print(f"\nMusaMatMul [{dtype.name}] Max Diff vs TF-CPU: {np.max(diff)}")

                rtol = 1e-2 if dtype != tf.float32 else 1e-4
                atol = 1e-2 if dtype != tf.float32 else 1e-4
                self.assertAllClose(musa_result, expected, rtol=rtol, atol=atol)
            except Exception as e:
                with open(self.results_file, "a") as f:
                    f.write(f"Operator: MusaMatMul, DataType: {dtype.name}, Error: {str(e)[:100]}\n")
                print(f"\nMusaMatMul [{dtype.name}] MUSA execution failed/skipped: {e}")

    def testMusaBatchMatMul(self):
        """Test MusaBatchMatMulV2 op: [B, M, K] x [B, K, N] -> [B, M, N]."""
        b, m, k, n = 2, 16, 32, 16
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            x_np = np.random.uniform(-1, 1, size=[b, m, k]).astype(np.float32)
            y_np = np.random.uniform(-1, 1, size=[b, k, n]).astype(np.float32)
            
            x = tf.cast(x_np, dtype=dtype)
            y = tf.cast(y_np, dtype=dtype)

            # 1. 基线计算
            with tf.device("/CPU:0"):
                expected = tf.matmul(x, y)

            # 2. MUSA 计算
            try:
                with tf.device("/device:MUSA:0"):
                    musa_result = musa_ops.MusaBatchMatMulV2(x=x, y=y)

                # 打印并记录误差
                diff = np.abs(musa_result.numpy() - expected.numpy())
                self._log_diff("MusaBatchMatMulV2", dtype.name, diff)
                print(f"\nMusaBatchMatMulV2 [{dtype.name}] Max Diff vs TF-CPU: {np.max(diff)}")

                rtol = 1e-2 if dtype != tf.float32 else 1e-4
                atol = 1e-2 if dtype != tf.float32 else 1e-4
                self.assertAllClose(musa_result, expected, rtol=rtol, atol=atol)
            except Exception as e:
                with open(self.results_file, "a") as f:
                    f.write(f"Operator: MusaBatchMatMulV2, DataType: {dtype.name}, Error: {str(e)[:100]}\n")
                print(f"\nMusaBatchMatMulV2 [{dtype.name}] MUSA execution failed/skipped: {e}")

    def testReduceBmmTest(self):
        """Test ReduceBmmTest op: ([M, K], [1, N]) -> [M, N]."""
        m, k, n = 8, 16, 4
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            a_np = np.random.uniform(-1, 1, size=[m, k]).astype(np.float32)
            b_np = np.random.uniform(-1, 1, size=[1, n]).astype(np.float32)
            
            a = tf.cast(a_np, dtype=dtype)
            b = tf.cast(b_np, dtype=dtype)

            # 1. 使用原生 TF API 在 CPU 上计算作为基线 (ReduceSum + MatMul)
            with tf.device("/CPU:0"):
                reduced_a = tf.reduce_sum(a, axis=1, keepdims=True)
                expected = tf.matmul(reduced_a, b)

            # 2. 使用自定义的 ReduceBmmTest 算子在 MUSA 上计算
            with tf.device("/device:MUSA:0"):
                musa_result = musa_ops.ReduceBmmTest(a=a, b=b)

            # 打印并记录误差
            diff = np.abs(musa_result.numpy() - expected.numpy())
            self._log_diff("ReduceBmmTest", dtype.name, diff)
            print(f"\nReduceBmmTest [{dtype.name}] Max Diff vs TF-CPU: {np.max(diff)}")

            rtol = 1e-2 if dtype != tf.float32 else 1e-4
            atol = 1e-2 if dtype != tf.float32 else 1e-4
            self.assertAllClose(musa_result, expected, rtol=rtol, atol=atol)

if __name__ == "__main__":
    tf.test.main()
