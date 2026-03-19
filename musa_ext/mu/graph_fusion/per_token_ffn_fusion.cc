#include "mu/graph_fusion/per_token_ffn_fusion.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

// Helper to check if node has specific op type
bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

// Helper to find node's input producer
const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  if (input.empty()) return nullptr;

  std::string node_name = input;
  if (node_name[0] == '^') {
    node_name = node_name.substr(1);
  }
  size_t colon_pos = node_name.find(':');
  if (colon_pos != std::string::npos) {
    node_name = node_name.substr(0, colon_pos);
  }

  for (int i = 0; i < graph.node_size(); ++i) {
    if (graph.node(i).name() == node_name) {
      return &graph.node(i);
    }
  }
  return nullptr;
}

}  // namespace

// =============================================================================
// MusaPerTokenFFNFusion Implementation
// =============================================================================

bool MusaPerTokenFFNFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow