"""Regression tests for plain TensorFlow ops lowered through MUSA XLA."""

import os

os.environ.setdefault("MUSA_ENABLE_TF32", "0")

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class XlaMusaComponentOpsTest(MUSATestCase):
    def run_xla(self, fn, *args):
        @tf.function(jit_compile=True)
        def compiled(*inputs):
            with tf.device("/device:MUSA:0"):
                return fn(*inputs)

        return compiled(*args)

    def test_reduce_sum_last_dim_matches_cpu(self):
        rng = np.random.RandomState(20)
        x = tf.constant(rng.standard_normal((4, 6, 8)).astype(np.float32))

        with tf.device("/CPU:0"):
            expected = tf.reduce_sum(x, axis=-1, keepdims=True)
        actual = self.run_xla(
            lambda a: tf.reduce_sum(a, axis=-1, keepdims=True), x
        )

        self.assertAllClose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_layer_norm_formula_matches_cpu(self):
        rng = np.random.RandomState(5)
        x = tf.constant(rng.standard_normal((4, 6, 8)).astype(np.float32))
        gamma = tf.constant(rng.uniform(0.5, 1.5, size=(8,)).astype(np.float32))
        beta = tf.constant(rng.uniform(-0.3, 0.3, size=(8,)).astype(np.float32))
        epsilon = 1e-5

        def layer_norm(a, scale, shift):
            mean = tf.reduce_mean(a, axis=-1, keepdims=True)
            variance = tf.reduce_mean(tf.square(a - mean), axis=-1, keepdims=True)
            return (a - mean) * tf.math.rsqrt(variance + epsilon) * scale + shift

        with tf.device("/CPU:0"):
            expected = layer_norm(x, gamma, beta)
        actual = self.run_xla(layer_norm, x, gamma, beta)

        self.assertAllClose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_matmul_2d_matches_cpu(self):
        rng = np.random.RandomState(10)
        lhs = tf.constant(rng.standard_normal((24, 6)).astype(np.float32))
        rhs = tf.constant(rng.standard_normal((6, 5)).astype(np.float32))

        with tf.device("/CPU:0"):
            expected = tf.matmul(lhs, rhs)
        actual = self.run_xla(lambda a, b: tf.matmul(a, b), lhs, rhs)

        self.assertAllClose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    tf.test.main()
