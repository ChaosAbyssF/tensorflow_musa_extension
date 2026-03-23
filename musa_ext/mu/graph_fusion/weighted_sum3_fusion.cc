#include "weighted_sum3_fusion.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

/*

input0 -> Mul0 \
                Add0 -> Add1 -> output
input1 -> Mul1 /       /
                      /
input2 -> Mul2 ------/

*/

namespace {
bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

bool IsAddOp(const NodeDef& node) {
  return IsOp(node, "Add") || IsOp(node, "AddV2") || IsOp(node, "BiasAdd");
}

// Helper to find node's input producer
const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  if (input.empty()) return nullptr;

  std::string node_name = input;
  if (node_name[0] == '^') {
    node_name = node_name.substr(1);
  }
  const size_t colon_pos = node_name.find(':');
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

bool HasOriginalSuffix(const std::string& node_name) {
  static const std::string kOriginalSuffix = "_original";
  return node_name.size() >= kOriginalSuffix.size() &&
         node_name.compare(node_name.size() - kOriginalSuffix.size(),
                           kOriginalSuffix.size(), kOriginalSuffix) == 0;
}

}  // namespace

bool MusaWeightedSum3Fusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

// Provide default constructor implementation to ensure symbol is exported
MusaWeightedSum3Fusion::MusaWeightedSum3Fusion() = default;

FusionMatchResult MusaWeightedSum3Fusion::Match(const GraphDef& graph,
                                                int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& add_node1 = graph.node(start_node_idx);

  // match start with Add node1
  if (!IsAddOp(add_node1)) {
    return result;
  }

  // find Add node0 & Mul node2
  const NodeDef* add_node0 = nullptr;
  const NodeDef* mul_node2 = nullptr;
  for (int i = 0; i < add_node1.input_size(); ++i) {
    const NodeDef* input_node = FindProducer(graph, add_node1.input(i));
    if (input_node && IsAddOp(*input_node)) {
      add_node0 = input_node;
    }
    if (input_node && IsOp(*input_node, "Mul")) {
      mul_node2 = input_node;
    }
  }

  if (!add_node0 || !mul_node2) {
    return result;
  }

  // find Mul node0 & Mul node1
  const NodeDef* mul_node0 = nullptr;
  const NodeDef* mul_node1 = nullptr;
  for (int i = 0; i < add_node0->input_size(); ++i) {
    const NodeDef* input_node = FindProducer(graph, add_node0->input(i));
    if (input_node && IsOp(*input_node, "Mul")) {
      if (!mul_node0) {
        mul_node0 = input_node;
      } else if (!mul_node1) {
        mul_node1 = input_node;
      }
    }
  }

  if (!mul_node0 || !mul_node1) {
    return result;
  }

  result.matched = true;
  result.matched_nodes.push_back(&add_node1);
  result.matched_nodes.push_back(add_node0);
  result.matched_nodes.push_back(mul_node2);
  result.matched_nodes.push_back(mul_node0);
  result.matched_nodes.push_back(mul_node1);

  result.captured_nodes["add1"] = &add_node1;
  result.captured_nodes["add0"] = add_node0;
  result.captured_nodes["mul2"] = mul_node2;
  result.captured_nodes["mul0"] = mul_node0;
  result.captured_nodes["mul1"] = mul_node1;

  return result;
}

Status MusaWeightedSum3Fusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid match result for MusaWeightedSum3Fusion");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  // Get captured nodes
  auto output_it = match_result.captured_nodes.find("add1");
  auto add0_it = match_result.captured_nodes.find("add0");
  auto mul2_it = match_result.captured_nodes.find("mul2");
  auto mul0_it = match_result.captured_nodes.find("mul0");
  auto mul1_it = match_result.captured_nodes.find("mul1");

  if (output_it == match_result.captured_nodes.end() ||
      add0_it == match_result.captured_nodes.end() ||
      mul2_it == match_result.captured_nodes.end() ||
      mul0_it == match_result.captured_nodes.end() ||
      mul1_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing captured nodes for MusaWeightedSum3Fusion");
  }

  const NodeDef* output_node = output_it->second;
  const NodeDef* add0_node = add0_it->second;
  const NodeDef* mul2_node = mul2_it->second;
  const NodeDef* mul0_node = mul0_it->second;
  const NodeDef* mul1_node = mul1_it->second;

  const std::string original_name = output_node->name();

  // Check if this output node has already been fused (avoid duplicates)
  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaWeightedSum3") {
      VLOG(1) << "MusaWeightedSum3Fusion: Output node " << original_name
              << " is already a fused node, skipping";
      return Status::OK();
    }
  }

  // Preserve output device before removing nodes
  const std::string output_device = output_node->device();

  VLOG(1) << "MusaWeightedSum3Fusion: Replacing " << original_name
          << " with MusaWeightedSum3";

  // Determine data and weight inputs for each Mul and set fused inputs.
  // MusaWeightedSum3 expects inputs: a, b, c, alpha, beta, gamma
  auto pick_data_and_weight = [&](const NodeDef* mul_node,
                                  std::string* data_input,
                                  std::string* weight_input) {
    if (!mul_node) return;
    if (mul_node->input_size() < 2) return;
    const std::string in0 = mul_node->input(0);
    const std::string in1 = mul_node->input(1);
    const NodeDef* p0 = FindProducer(*graph, in0);
    const NodeDef* p1 = FindProducer(*graph, in1);
    // Prefer treating a Const producer as the weight (scalar)
    if (p0 && p0->op() == "Const") {
      *weight_input = in0;
      *data_input = in1;
    } else if (p1 && p1->op() == "Const") {
      *weight_input = in1;
      *data_input = in0;
    } else {
      // Fallback: assume input(1) is weight (original code used input(0) as
      // data)
      *data_input = in0;
      *weight_input = in1;
    }
  };

  std::string data0, w0, data1, w1, data2, w2;
  pick_data_and_weight(mul0_node, &data0, &w0);
  pick_data_and_weight(mul1_node, &data1, &w1);
  pick_data_and_weight(mul2_node, &data2, &w2);

  // data/weight inputs will be appended after fused node is created

  // Collect nodes that belong to the matched subgraph and will be removed.
  std::unordered_set<std::string> fuse_node_names = {
      original_name, add0_node->name(), mul0_node->name(), mul1_node->name(),
      mul2_node->name()};

  VLOG(2) << "MusaWeightedSum3Fusion: will remove " << fuse_node_names.size()
          << " nodes";

  // Create fused node (with the original output name)
  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name +
                       "_fused");  // Temporary name until original is removed
  fused_node->set_op("MusaWeightedSum3");
  fused_node->set_device(output_device);

  // data inputs
  if (!data0.empty()) fused_node->add_input(data0);
  if (!data1.empty()) fused_node->add_input(data1);
  if (!data2.empty()) fused_node->add_input(data2);

  // weight (scalar) inputs: alpha, beta, gamma
  if (!w0.empty()) fused_node->add_input(w0);
  if (!w1.empty()) fused_node->add_input(w1);
  if (!w2.empty()) fused_node->add_input(w2);

  // Redirect consumers of the original output node to the new fused node
  FusionGraphUtils::RedirectInputs(graph, original_name, fused_node->name());

  // Now safely remove original nodes.
  int removed_count = 0;
  for (const auto& node_name : fuse_node_names) {
    int idx = FusionGraphUtils::FindNodeIndex(*graph, node_name);
    if (idx >= 0) {
      VLOG(2) << "MusaWeightedSum3Fusion: Removing node: " << node_name;
      FusionGraphUtils::RemoveNode(graph, idx);
      removed_count++;
    }
  }

  // Restore original name to the fused node
  fused_node->set_name(original_name);

  VLOG(2) << "MusaWeightedSum3Fusion: removed " << removed_count << " nodes";
  if (!w0.empty()) fused_node->add_input(w0);
  if (!w1.empty()) fused_node->add_input(w1);
  if (!w2.empty()) fused_node->add_input(w2);

  // Try to remove weight Const nodes (alpha/beta/gamma) if they are unused.
  // The fused inputs (w0/w1/w2) may contain port suffixes; get producer names.
  std::vector<std::string> weight_candidates;
  auto get_prod = [](const std::string& input) {
    if (input.empty()) return std::string();
    // Strip control prefix and port
    if (input[0] == '^') return input.substr(1);
    size_t colon = input.find(':');
    if (colon != std::string::npos) return input.substr(0, colon);
    return input;
  };
  if (!w0.empty()) weight_candidates.push_back(get_prod(w0));
  if (!w1.empty()) weight_candidates.push_back(get_prod(w1));
  if (!w2.empty()) weight_candidates.push_back(get_prod(w2));

  if (!weight_candidates.empty()) {
    FusionGraphUtils::RemoveNodesIfUnused(graph, weight_candidates, {});
  }

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaWeightedSum3Fusion);

REGISTER_FUSION_KERNEL(MusaWeightedSum3Fusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
