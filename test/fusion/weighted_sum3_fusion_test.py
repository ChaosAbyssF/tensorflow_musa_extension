# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""End-to-end tests for MusaWeightedSum3 fusion.

Pattern tested (simple form):

  input0 -> Mul0 \
                  Add0 -> Add1 -> output
  input1 -> Mul1 /       /       /
                        /       /
  input2 -> Mul2 ----/       /

The fused op should be `MusaWeightedSum3` with inputs (a, b, c, alpha, beta, gamma).
"""

import os
import tempfile
import glob

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from musa_test_utils import MUSATestCase
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2


def _load_last_after_fusion_pbtxt(dump_dir):
    """Load the last *_after_fusion.pbtxt from `dump_dir`.

    Returns (text, GraphDef).
    """
    dump_files = sorted(glob.glob(os.path.join(dump_dir, "*_after_fusion.pbtxt")))
    if not dump_files:
        raise RuntimeError(f"No after_fusion dump found in {dump_dir}")

    with open(dump_files[-1], "r", encoding="utf-8") as handle:
        dump_text = handle.read()

    graph_def = graph_pb2.GraphDef()
    text_format.Parse(dump_text, graph_def)
    return dump_text, graph_def


def create_config_with_musa_optimizer():
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rw = config.graph_options.rewrite_options
    rw.min_graph_nodes = -1

    custom_opt = rw.custom_optimizers.add()
    custom_opt.name = "musa_graph_optimizer"
    rw.optimizers.extend(["musa_graph_optimizer"])

    return config


class WeightedSum3FusionTest(MUSATestCase):
    """Tests for MusaWeightedSum3 fusion."""

    def _build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                a = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name="a")
                b = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name="b")
                c = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name="c")

                w0 = tf.constant(np.array([0.5], dtype=np.float32), name="w0")
                w1 = tf.constant(np.array([1.5], dtype=np.float32), name="w1")
                w2 = tf.constant(np.array([2.0], dtype=np.float32), name="w2")

                mul0 = tf.multiply(a, w0, name="mul0")
                mul1 = tf.multiply(b, w1, name="mul1")
                add0 = tf.add(mul0, mul1, name="add0")
                mul2 = tf.multiply(c, w2, name="mul2")
                out = tf.add(add0, mul2, name="add1")

        return graph, a, b, c, out

    def test_fusion_is_applied(self):
        """The optimized graph must contain a MusaWeightedSum3 node."""
        graph, a, b, c, out = self._build_graph()

        rng = np.random.RandomState(7)
        batch = 2
        a_np = rng.standard_normal((batch, 4)).astype(np.float32)
        b_np = rng.standard_normal((batch, 4)).astype(np.float32)
        c_np = rng.standard_normal((batch, 4)).astype(np.float32)

        config = create_config_with_musa_optimizer()
        # Use GraphDef dump produced by the optimizer to verify fusion (some
        # execution partitions may not reflect the fused op; the optimizer
        # writes the after-fusion GraphDef which we can inspect).
        old_dump = os.environ.get("MUSA_DUMP_GRAPHDEF")
        old_dump_dir = os.environ.get("MUSA_DUMP_GRAPHDEF_DIR")

        with tempfile.TemporaryDirectory(prefix="musa_weighted_sum3_") as dump_dir:
            os.environ["MUSA_DUMP_GRAPHDEF"] = "1"
            os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = dump_dir

            try:
                with tf.compat.v1.Session(graph=graph, config=config) as sess:
                    result = sess.run(out, feed_dict={a: a_np, b: b_np, c: c_np})
            finally:
                if old_dump is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF"] = old_dump

                if old_dump_dir is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF_DIR", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = old_dump_dir

            dump_text, gd = _load_last_after_fusion_pbtxt(dump_dir)

            # Numerical sanity check
            expected = 0.5 * a_np + 1.5 * b_np + 2.0 * c_np
            self.assertAllClose(result, expected, rtol=1e-6, atol=1e-6)

            # Verify fused node exists in dumped GraphDef
            self.assertIn('op: "MusaWeightedSum3"', dump_text)


    def test_fusion_and_numerical_correctness(self):
        """Fusion appears and fused op computes same result as plain ops."""
        rng = np.random.RandomState(123)
        batch = 3
        a_np = rng.standard_normal((batch, 4)).astype(np.float32)
        b_np = rng.standard_normal((batch, 4)).astype(np.float32)
        c_np = rng.standard_normal((batch, 4)).astype(np.float32)

        graph, a, b, c, out = self._build_graph()

        expected = 0.5 * a_np + 1.5 * b_np + 2.0 * c_np

        config = create_config_with_musa_optimizer()

        # Use GraphDef dump to inspect the optimizer output
        old_dump = os.environ.get("MUSA_DUMP_GRAPHDEF")
        old_dump_dir = os.environ.get("MUSA_DUMP_GRAPHDEF_DIR")

        with tempfile.TemporaryDirectory(prefix="musa_weighted_sum3_") as dump_dir:
            os.environ["MUSA_DUMP_GRAPHDEF"] = "1"
            os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = dump_dir

            try:
                with tf.compat.v1.Session(graph=graph, config=config) as sess:
                    result = sess.run(out, feed_dict={a: a_np, b: b_np, c: c_np})
            finally:
                if old_dump is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF"] = old_dump

                if old_dump_dir is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF_DIR", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = old_dump_dir

            dump_text, gd = _load_last_after_fusion_pbtxt(dump_dir)

            # 1) Fusion was applied
            self.assertIn('op: "MusaWeightedSum3"', dump_text)

            fused_nodes = [n for n in gd.node if n.op == "MusaWeightedSum3"]
            self.assertEqual(len(fused_nodes), 1, "Expected exactly one MusaWeightedSum3 node")

            fused = fused_nodes[0]
            # MusaWeightedSum3 expects 6 inputs: a, b, c, alpha, beta, gamma
            self.assertEqual(len(fused.input), 6, f"Fused node inputs: {fused.input}")

            # 2) No residual original nodes left
            residual_originals = [n.name for n in gd.node if n.name.endswith("_original")]
            self.assertFalse(residual_originals, f"Residual original nodes: {residual_originals}")

            # 3) Helper Mul/Add nodes should be removed (not present by name)
            helper_names = {"mul0", "mul1", "mul2", "add0"}
            remaining_helpers = [n.name for n in gd.node if n.name in helper_names]
            self.assertFalse(remaining_helpers, f"Helper nodes remaining: {remaining_helpers}")

        # Numerical check
        self.assertAllClose(result, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tf.test.main()
