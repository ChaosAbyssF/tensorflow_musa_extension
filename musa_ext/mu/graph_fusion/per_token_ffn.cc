#include "mu/graph_fusion/per_token_ffn.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/logging.h"

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
}  // namespace

// Small helper to push node into vector if not already present.
static void PushUnique(std::vector<const NodeDef*>* nodes,
                       const NodeDef* node) {
  if (!node) return;
  auto it = std::find(nodes->begin(), nodes->end(), node);
  if (it == nodes->end()) nodes->push_back(node);
}

// Helper: try to find an underlying BatchMatMul producer by walking backwards
// from a node through a small set of elementwise / activation nodes. If
// successful, returns true and sets *out_bm and appends traversed nodes to
// `chain` (excluding the final BatchMatMul node).
static bool TryFindBatchMatMul(const GraphDef& graph, const NodeDef* start,
                               const NodeDef** out_bm,
                               std::vector<const NodeDef*>* chain) {
  if (!start) return false;

  std::vector<const NodeDef*> stack;
  std::unordered_set<std::string> visited;
  stack.push_back(start);

  int max_steps = 12;
  while (!stack.empty() && max_steps-- > 0) {
    const NodeDef* node = stack.back();
    stack.pop_back();
    if (!node) continue;
    if (visited.insert(node->name()).second == false) continue;

    if (IsOp(*node, "BatchMatMulV2") || IsOp(*node, "BatchMatMul") ||
        IsOp(*node, "MatMul")) {
      *out_bm = node;
      return true;
    }

    // Record node as part of chain if it's an intermediate op we allow
    chain->push_back(node);

    // Explore its producers (non-const)
    for (int i = 0; i < node->input_size(); ++i) {
      const NodeDef* p = FindProducer(graph, node->input(i));
      if (!p) continue;
      // Skip constants
      if (p->op() == "Const") continue;
      // Allow traversing through common elementwise / activation ops
      if (IsOp(*p, "Transpose") || IsOp(*p, "Mul") || IsOp(*p, "Add") ||
          IsOp(*p, "AddV2") || IsOp(*p, "BiasAdd") || IsOp(*p, "Erf") ||
          IsOp(*p, "Erfc") || IsOp(*p, "RealDiv") || IsOp(*p, "Div") ||
          IsOp(*p, "Tanh") || IsOp(*p, "Pow") || IsOp(*p, "Neg") ||
          IsOp(*p, "Sqrt") || IsOp(*p, "BatchMatMulV2") ||
          IsOp(*p, "BatchMatMul") || IsOp(*p, "MatMul")) {
        stack.push_back(p);
      }
    }
  }

  return false;
}

bool MusaPerTokenFFNFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaPerTokenFFNFusion::Match(const GraphDef& graph,
                                               int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& start_node = graph.node(start_node_idx);

  // start must be an Add-like node
  if (!IsOp(start_node, "Add") && !IsOp(start_node, "AddV2") &&
      !IsOp(start_node, "BiasAdd")) {
    return result;
  }

  // Avoid matching nodes that already were replaced
  auto has_original_suffix = [](const std::string& node_name) {
    static const std::string kOriginalSuffix = "_original";
    return node_name.size() >= kOriginalSuffix.size() &&
           node_name.compare(node_name.size() - kOriginalSuffix.size(),
                             kOriginalSuffix.size(), kOriginalSuffix) == 0;
  };
  if (has_original_suffix(start_node.name())) return result;

  // Try to find a MatMul/BMM that produces the start node's output (bm1),
  // then walk upstream from bm1 to find an earlier MatMul/BMM (bm0)
  const NodeDef* bm0 = nullptr;
  const NodeDef* bm1 = nullptr;
  std::vector<const NodeDef*> chain0;
  std::vector<const NodeDef*> chain1;

  // Find bm1: a MatMul/BatchMatMul producer feeding the output Add/BiasAdd
  for (int i = 0; i < start_node.input_size(); ++i) {
    const NodeDef* inp = FindProducer(graph, start_node.input(i));
    if (!inp) continue;
    const NodeDef* found_bm = nullptr;
    std::vector<const NodeDef*> chain;
    if (TryFindBatchMatMul(graph, inp, &found_bm, &chain)) {
      bm1 = found_bm;
      chain1 = chain;
      break;
    }
  }

  if (!bm1) return result;

  // Find bm0 by looking at bm1's inputs (the activation branch)
  for (int i = 0; i < bm1->input_size(); ++i) {
    const NodeDef* inp = FindProducer(graph, bm1->input(i));
    if (!inp) continue;
    const NodeDef* found_bm = nullptr;
    std::vector<const NodeDef*> chain;
    if (TryFindBatchMatMul(graph, inp, &found_bm, &chain)) {
      if (found_bm != bm1) {
        bm0 = found_bm;
        chain0 = chain;
        break;
      }
    }
  }

  if (!bm0) return result;

  // Build matched nodes list: include start node, both matmuls and intermediate
  // chains
  result.matched = true;
  result.matched_nodes.push_back(&start_node);
  result.matched_nodes.push_back(bm0);
  result.matched_nodes.push_back(bm1);
  for (const NodeDef* n : chain0) PushUnique(&result.matched_nodes, n);
  for (const NodeDef* n : chain1) PushUnique(&result.matched_nodes, n);

  result.captured_nodes["output"] = &start_node;
  result.captured_nodes["batch_matmul_0"] = bm0;
  result.captured_nodes["batch_matmul_1"] = bm1;

  VLOG(INFO) << "Matched PerTokenFFN pattern with output node: "
             << start_node.name() << ", batch_matmul_0: " << bm0->name()
             << ", batch_matmul_1: " << bm1->name();

  return result;
}

tensorflow::Status MusaPerTokenFFNFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return tensorflow::Status(error::INVALID_ARGUMENT,
                              "Invalid PerTokenFFN match result");
  }

  if (!IsKernelAvailable()) return tensorflow::Status::OK();

  auto out_it = match_result.captured_nodes.find("output");
  auto bm0_it = match_result.captured_nodes.find("batch_matmul_0");
  auto bm1_it = match_result.captured_nodes.find("batch_matmul_1");

  if (out_it == match_result.captured_nodes.end() ||
      bm0_it == match_result.captured_nodes.end() ||
      bm1_it == match_result.captured_nodes.end()) {
    return tensorflow::Status(error::INVALID_ARGUMENT,
                              "Missing required nodes in PerTokenFFN pattern");
  }

  const NodeDef* output_node = out_it->second;
  const NodeDef* bm0 = bm0_it->second;
  const NodeDef* bm1 = bm1_it->second;

  const std::string original_name = output_node->name();
  const std::string original_output_name = original_name + "_original";

  // Avoid duplicate fusion
  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaPerTokenFFN") {
      return tensorflow::Status::OK();
    }
  }

  int output_idx = FusionGraphUtils::FindNodeIndex(*graph, original_name);
  if (output_idx < 0) {
    return tensorflow::Status(
        error::INVALID_ARGUMENT,
        "Failed to find output node in graph: " + original_name);
  }

  NodeDef* original_output_node = graph->mutable_node(output_idx);
  const std::string output_device = original_output_node->device();

  // Preserve dtype if present
  AttrValue output_dtype;
  const auto dtype_it = original_output_node->attr().find("T");
  const bool has_output_dtype = dtype_it != original_output_node->attr().end();
  if (has_output_dtype) output_dtype = dtype_it->second;

  // Rename original node
  original_output_node->set_name(original_output_name);

  // Create fused node with same original name
  NodeDef* fused = graph->add_node();
  fused->set_name(original_name);
  fused->set_op("MusaPerTokenFFN");
  fused->set_device(output_device);

  // Set inputs to fused node: use the original inputs of the output node so
  // the fused op receives same upstream tensors (conservative mapping).
  for (int i = 0; i < original_output_node->input_size(); ++i) {
    fused->add_input(original_output_node->input(i));
  }

  if (has_output_dtype) {
    (*fused->mutable_attr())["T"] = output_dtype;
  } else {
    (*fused->mutable_attr())["T"].set_type(DT_FLOAT);
  }

  // Remove matched nodes if unused. Protect inputs to fused node.
  std::vector<std::string> removable;
  removable.reserve(match_result.matched_nodes.size());
  for (const NodeDef* n : match_result.matched_nodes) {
    if (!n) continue;
    if (n->name() == original_name) continue;
    if (n->name() == original_output_name) continue;
    removable.push_back(n->name());
  }

  // Protected: any input names used by fused node and the fused node itself
  std::unordered_set<std::string> protected_names;
  for (int i = 0; i < fused->input_size(); ++i) {
    protected_names.insert(
        FusionGraphUtils::GetProducerNodeName(fused->input(i)));
  }
  protected_names.insert(original_name);

  FusionGraphUtils::RemoveNodesIfUnused(graph, removable, protected_names);

  VLOG(INFO) << "Applied PerTokenFFN fusion, removed " << removable.size()
             << " nodes";

  return tensorflow::Status::OK();
}

// Register the pattern
REGISTER_FUSION_PATTERN(MusaPerTokenFFNFusion);
// Register kernel availability
REGISTER_FUSION_KERNEL(MusaPerTokenFFNFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

// (Duplicate Apply implementation removed — the file contains a single
// tensorflow::Status-qualified Apply implementation earlier.)
