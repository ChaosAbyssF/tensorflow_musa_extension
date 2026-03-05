# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for the MUSA IsNonZeroCount operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class IsNonZeroCountOpTest(MUSATestCase):
  """Coverage for the MUSA IsNonZeroCount kernel."""

  _DTYPES = [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64, tf.bool]

  def _numpy_dtype(self, dtype):
    if dtype == tf.bfloat16:
      return np.float32
    return dtype.as_numpy_dtype

  def _run_musa_count(self, values, dtype):
    numpy_values = values.astype(self._numpy_dtype(dtype))
    with tf.device('/device:MUSA:0'):
      tensor = tf.constant(numpy_values, dtype=dtype)
      result = tf.raw_ops.MusaIsNonZeroCount(input=tensor)
    return int(result.numpy())

  def testZeroTensor(self):
    """Every dtype should return 0 when the tensor is all zeros."""
    shape = (4, 3, 2)
    for dtype in self._DTYPES:
      zeros = np.zeros(shape, dtype=self._numpy_dtype(dtype))
      self.assertEqual(self._run_musa_count(zeros, dtype), 0)

  def testRandomTensor(self):
    """Verify MUSA agrees with NumPy when the tensor contains random values."""
    rng = np.random.default_rng(1234)
    shape = (6, 5)

    for dtype in self._DTYPES:
      if dtype == tf.bool:
        data = rng.choice([False, True], size=shape)
      elif dtype in (tf.int32, tf.int64):
        data = rng.integers(-5, 6, size=shape).astype(self._numpy_dtype(dtype))
      else:
        data = rng.uniform(-10, 10, size=shape).astype(self._numpy_dtype(dtype))

      expected = np.count_nonzero(data)
      actual = self._run_musa_count(data, dtype)
      self.assertEqual(actual, int(expected))


if __name__ == "__main__":
  tf.test.main()
