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

    def testReduceTest(self):
        """Test ReduceTest op: [M, K] -> [M] (sum along dim 1)."""
        shape = [16, 32]
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            x_np = np.random.uniform(-1, 1, size=shape).astype(np.float32)
            x = tf.cast(x_np, dtype=dtype)

            # Define expected function (Sum along axis 1)
            def ref_func(input_tensor):
                return tf.reduce_sum(input_tensor, axis=1)

            # Custom Op call
            def musa_op_func(input_tensor):
                return musa_ops.ReduceTest(input=input_tensor)

            # Use standard tolerances as in other test files
            rtol = 1e-2 if dtype != tf.float32 else 1e-5
            atol = 1e-2 if dtype != tf.float32 else 1e-5
            
            # Since these are custom ops, _compare_cpu_musa_results might need direct function calls
            # We assume MUSATestCase handles raw_ops or we use the custom function wrapper
            self._compare_cpu_musa_results(musa_op_func, [x], dtype=dtype, rtol=rtol, atol=atol)

    def testReduceBmmTest(self):
        """Test ReduceBmmTest op: ([M, K], [1, N]) -> [M, N]."""
        m, k, n = 8, 16, 4
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            a_np = np.random.uniform(-1, 1, size=[m, k]).astype(np.float32)
            b_np = np.random.uniform(-1, 1, size=[1, n]).astype(np.float32)
            
            a = tf.cast(a_np, dtype=dtype)
            b = tf.cast(b_np, dtype=dtype)

            def ref_func(in_a, in_b):
                reduced_a = tf.reduce_sum(in_a, axis=1, keepdims=True)
                return tf.matmul(reduced_a, in_b)

            def musa_op_func(in_a, in_b):
                return musa_ops.ReduceBmmTest(a=in_a, b=in_b)

            rtol = 1e-2 if dtype != tf.float32 else 1e-5
            atol = 1e-2 if dtype != tf.float32 else 1e-5

            self._compare_cpu_musa_results(musa_op_func, [a, b], dtype=dtype, rtol=rtol, atol=atol)

if __name__ == "__main__":
    tf.test.main()
