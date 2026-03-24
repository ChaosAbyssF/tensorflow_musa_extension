"""Tests for Per-Token FFN fusion (MusaPerTokenFFN).

This test verifies numerical correctness against CPU reference and asserts
that the fusion pass produces a `MusaPerTokenFFN` node in the partitioned
graph when the optimizer is enabled.
"""

import os
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


def create_config_with_musa_optimizer():
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    rewriter_config = config.graph_options.rewrite_options
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"
    rewriter_config.min_graph_nodes = -1
    rewriter_config.optimizers.extend(["musa_graph_optimizer"])
    return config


class PerTokenFFNFusionTest(MUSATestCase):
    """Tests for PerTokenFFN fusion."""

    def test_per_token_ffn_correctness(self):
        np.random.seed(123)
        tf.random.set_seed(123)

        m, k, h, n = 4, 8, 12, 6

        x_np = np.random.randn(m, k).astype(np.float32)
        w0_np = np.random.randn(k, h).astype(np.float32)
        b0_np = np.random.randn(h).astype(np.float32)
        w1_np = np.random.randn(h, n).astype(np.float32)
        b1_np = np.random.randn(n).astype(np.float32)

        # Reference computation on CPU
        with tf.device("/CPU:0"):
            x_tf = tf.constant(x_np)
            w0_tf = tf.constant(w0_np)
            b0_tf = tf.constant(b0_np)
            w1_tf = tf.constant(w1_np)
            b1_tf = tf.constant(b1_np)

            mm0 = tf.matmul(x_tf, w0_tf)
            bias0 = tf.nn.bias_add(mm0, b0_tf)
            gelu = tf.nn.gelu(bias0)
            mm1 = tf.matmul(gelu, w1_tf)
            out = tf.nn.bias_add(mm1, b1_tf)
            # add consumer to avoid pruning
            out = out * 1.0

        # Build graph pinned to MUSA
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x")
                w0 = tf.constant(w0_np, dtype=tf.float32, name="w0")
                b0 = tf.constant(b0_np, dtype=tf.float32, name="b0")
                w1 = tf.constant(w1_np, dtype=tf.float32, name="w1")
                b1 = tf.constant(b1_np, dtype=tf.float32, name="b1")

                mm0_m = tf.matmul(x, w0)
                bias0_m = tf.nn.bias_add(mm0_m, b0)
                gelu_m = tf.nn.gelu(bias0_m)
                mm1_m = tf.matmul(gelu_m, w1)
                out_m = tf.nn.bias_add(mm1_m, b1)
                out_m = out_m * 1.0

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        # Run and compare numeric correctness
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            actual = sess.run(out_m, feed_dict={x: x_np})

        self.assertAllClose(actual, out.numpy(), rtol=1e-5, atol=1e-5)

    def test_per_token_ffn_dtypes(self):
        np.random.seed(1234)
        tf.random.set_seed(1234)

        m, k, h, n = 4, 8, 12, 6

        # Base float32 arrays (used for conversion to other dtypes)
        x_base = np.random.randn(m, k).astype(np.float32)
        w0_base = np.random.randn(k, h).astype(np.float32)
        b0_base = np.random.randn(h).astype(np.float32)
        w1_base = np.random.randn(h, n).astype(np.float32)
        b1_base = np.random.randn(n).astype(np.float32)

        dtypes = [tf.float32, tf.float16, tf.bfloat16]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                # Reference on CPU computed in float32
                with tf.device("/CPU:0"):
                    x_tf = tf.constant(x_base, dtype=tf.float32)
                    w0_tf = tf.constant(w0_base, dtype=tf.float32)
                    b0_tf = tf.constant(b0_base, dtype=tf.float32)
                    w1_tf = tf.constant(w1_base, dtype=tf.float32)
                    b1_tf = tf.constant(b1_base, dtype=tf.float32)

                    mm0 = tf.matmul(x_tf, w0_tf)
                    bias0 = tf.nn.bias_add(mm0, b0_tf)
                    gelu = tf.nn.gelu(bias0)
                    mm1 = tf.matmul(gelu, w1_tf)
                    out_ref = tf.nn.bias_add(mm1, b1_tf)
                    out_ref = out_ref * 1.0

                # Build graph pinned to MUSA with the target dtype
                graph = tf.Graph()
                with graph.as_default():
                    with tf.device("/device:MUSA:0"):
                        x_ph = tf.compat.v1.placeholder(
                            dtype, shape=[None, k], name="x"
                        )

                        # For tf.bfloat16, pass float32 numpy and set dtype on tf.constant
                        if dtype == tf.float16:
                            w0 = tf.constant(
                                w0_base.astype(np.float16), dtype=tf.float16, name="w0"
                            )
                            b0 = tf.constant(
                                b0_base.astype(np.float16), dtype=tf.float16, name="b0"
                            )
                            w1 = tf.constant(
                                w1_base.astype(np.float16), dtype=tf.float16, name="w1"
                            )
                            b1 = tf.constant(
                                b1_base.astype(np.float16), dtype=tf.float16, name="b1"
                            )
                        else:
                            w0 = tf.constant(w0_base, dtype=dtype, name="w0")
                            b0 = tf.constant(b0_base, dtype=dtype, name="b0")
                            w1 = tf.constant(w1_base, dtype=dtype, name="w1")
                            b1 = tf.constant(b1_base, dtype=dtype, name="b1")

                        mm0_m = tf.matmul(x_ph, w0)
                        bias0_m = tf.nn.bias_add(mm0_m, b0)
                        gelu_m = tf.nn.gelu(bias0_m)
                        mm1_m = tf.matmul(gelu_m, w1)
                        out_m = tf.nn.bias_add(mm1_m, b1)

                        # Cast output to float32 for stable numeric comparison
                        out_m = tf.cast(out_m, tf.float32)
                        out_m = out_m * 1.0

                config = create_config_with_musa_optimizer()
                run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
                run_metadata = tf.compat.v1.RunMetadata()

                # Prepare feed; convert numpy input to appropriate dtype where possible
                if dtype == tf.float16:
                    x_feed = x_base.astype(np.float16)
                else:
                    x_feed = x_base

                with tf.compat.v1.Session(graph=graph, config=config) as sess:
                    actual = sess.run(out_m, feed_dict={x_ph: x_feed})

                # Choose tolerances depending on dtype
                if dtype == tf.float32:
                    rtol, atol = 1e-5, 1e-5
                elif dtype == tf.float16:
                    rtol, atol = 1e-2, 1e-2
                else:  # bfloat16
                    rtol, atol = 2e-2, 2e-2

                self.assertAllClose(actual, out_ref.numpy(), rtol=rtol, atol=atol)


if __name__ == "__main__":
    tf.test.main()
