import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase
from tensorflow.core.protobuf import config_pb2

def create_config_with_musa_optimizer():
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    rewriter_config = config.graph_options.rewrite_options
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"
    rewriter_config.min_graph_nodes = -1
    return config

class RealDivAddFusionTest(MUSATestCase):
    def test_realdiv_add_fusion(self):
        print("\n" + "="*70)
        print("Test: RealDiv + Add Fusion")
        print("="*70)

        shape = [2, 1024]
        np.random.seed(42)
        x_np = np.random.randn(*shape).astype(np.float32)
        y_np = (np.random.rand(*shape) + 0.1).astype(np.float32)
        z_np = np.random.randn(*shape).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(tf.float32, shape=shape, name="x")
                y = tf.compat.v1.placeholder(tf.float32, shape=shape, name="y")
                z = tf.compat.v1.placeholder(tf.float32, shape=shape, name="z")
                
                div_out = tf.realdiv(x, y, name="my_div")
                output = tf.raw_ops.AddV2(x=div_out, y=z, name="my_add")

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            # 运行并验证结果
            # 注意：在真实的 MUSA 环境中，你可以通过设置环境变量 TF_CPP_MIN_VLOG_LEVEL=3 来观察融合是否发生
            result = sess.run(output, feed_dict={x: x_np, y: y_np, z: z_np})
            
            # 计算预期结果进行对比
            expected = (x_np / y_np) + z_np
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
            print("Numerical check: PASSED")

        print("\n" + "="*70)
        print("✓ RealDiv + Add Fusion test completed")

if __name__ == "__main__":
    tf.test.main()
