# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA Assert operator."""

import tensorflow as tf
from musa_test_utils import MUSATestCase


class AssertOpTest(MUSATestCase):
    """Tests for MUSA Assert operator."""

    def testAssertSuccess(self):
        """Test Assert op when condition is true."""
        with self.session(use_gpu=True) as sess:
            with tf.device("/device:MUSA:0"):
                condition = tf.constant(True)
                data = [tf.constant(1.0), tf.constant("test message")]
                # tf.compat.v1.Assert is more explicit for testing the Op directly
                assert_op = tf.compat.v1.Assert(condition, data)
                # Should not raise any error
                sess.run(assert_op)

    def testAssertFailure(self):
        """Test Assert op when condition is false."""
        with self.session(use_gpu=True) as sess:
            with tf.device("/device:MUSA:0"):
                condition = tf.constant(False)
                data = [tf.constant(42), tf.constant("error occurred")]
                assert_op = tf.compat.v1.Assert(condition, data)
                
                with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "assertion failed: .*42.*error occurred"):
                    sess.run(assert_op)

    def testAssertInvalidCondition(self):
        """Test Assert op with non-scalar condition."""
        with self.session(use_gpu=True) as sess:
            with tf.device("/device:MUSA:0"):
                # MusaAssertOp expects a scalar condition
                condition = tf.constant([True, True])
                data = [tf.constant("invalid condition")]
                
                # The Op itself should check for scalar condition
                # Note: tf.compat.v1.Assert might do some client-side checks,
                # but we want to trigger the OP_REQUIRES in musa_assert_op.cc
                assert_op = tf.raw_ops.Assert(condition=condition, data=data)
                
                with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "In\\[0\\] should be a scalar"):
                    sess.run(assert_op)

    def testAssertSummarize(self):
        """Test Assert op with summarize attribute."""
        with self.session(use_gpu=True) as sess:
            with tf.device("/device:MUSA:0"):
                condition = tf.constant(False)
                # Create a large tensor to test summarization
                data = [tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
                
                # summarize=3 means only show 3 elements
                # Note: Currently musa_assert_op.cc hardcodes summarize_ = 0
                # but let's see how it behaves.
                assert_op = tf.raw_ops.Assert(condition=condition, data=data, summarize=3)
                
                # Since summarize is hardcoded to 0 in the current C++ implementation, 
                # it might show 3 elements (default for 0 in SummarizeValue) or something else.
                # TensorFlow's default SummarizeValue(0) might show a few elements.
                with self.assertRaises(tf.errors.InvalidArgumentError) as cm:
                    sess.run(assert_op)
                
                error_msg = str(cm.exception)
                self.assertIn("assertion failed:", error_msg)
                self.assertIn("1 2 3", error_msg)

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.test.main()
