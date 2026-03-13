import os
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2

def check_fusion_count(pb_path, plugin_path):
    if not os.path.exists(pb_path):
        print(f"Error: File {pb_path} not found.")
        return

    # 加载已构建的 MUSA 插件库
    try:
        # 这是一个关键步骤，显式加载 .so 文件。
        # 这会触发插件内部的全局对象构造，完成算子和图优化器的注册。
        tf.load_op_library(plugin_path)
        print(f"Successfully loaded MUSA plugin: {plugin_path}")
    except Exception as e:
        print(f"Warning: Could not load MUSA plugin definitively: {e}")

    # 1. 加载 Frozen Graph
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    # 修改节点设备为 MUSA，否则 musa_graph_optimizer 会因为没有 MUSA 节点而跳过优化
    for node in graph_def.node:
        node.device = "/device:MUSA:0"
    
    # 2. 配置 MUSA 优化器运行 Session 以触发融合
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    rewriter_config = config.graph_options.rewrite_options
    
    # 显式添加 musa_graph_optimizer
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"
    
    # 修复参数设置：使用 bytes 字符串 "ON"
    parameter_map = custom_optimizer.parameter_map
    parameter_map["remapping"].s = b"ON"
    
    rewriter_config.min_graph_nodes = -1 # 强制触发优化

    # 构建一个临时的 Graph 用于执行优化
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session(config=config) as sess:
        # 将读取的 GraphDef 导入当前 Session 的 Graph
        tf.import_graph_def(graph_def, name="")
        
        # 获取经过优化器处理后的 GraphDef
        # TensorFlow 在 Session 运行时会根据配置执行 Grappler 优化
        optimized_graph = sess.graph_def
        
        # 3. 统计融合后的算子数量
        fused_op_name = "MusafusedRealDivAdd"
        fused_nodes = [n for n in optimized_graph.node if n.op == fused_op_name]
        
        print(f"Optimized graph node count: {len(optimized_graph.node)}")
        print(f"Number of '{fused_op_name}' nodes created: {len(fused_nodes)}")
        
        if len(fused_nodes) > 0:
            print("\nExample fused nodes:")
            for i, node in enumerate(fused_nodes[:5]):
                print(f"  {i+1}. {node.name}")
        else:
            print("\nNo fusion performed. Ensure 'musa_plugin' is correctly loaded and the device is available.")

if __name__ == "__main__":
    PB_FILE = "/workspace/tf_graph/meta_graph_3_frozen.pb"
    PLUGIN_PATH = "/workspace/tensorflow_musa_extension/build/libmusa_plugin.so"
    check_fusion_count(PB_FILE, PLUGIN_PATH)
