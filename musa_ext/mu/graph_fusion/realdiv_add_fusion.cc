#include "mu/graph_fusion/realdiv_add_fusion.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

FusionMatchResult RealDivAddFusion::Match(const GraphDef& graph,
                                         int start_node_idx) const {
  const NodeDef* node = &graph.node(start_node_idx);
  if (node->op() != "AddV2") return FusionMatchResult{};

  FusionMatchResult result;
  
  // Pattern: RealDiv -> AddV2
  // We start from AddV2 and look for RealDiv as input
  for (const std::string& input : node->input()) {
    std::string producer_name = FusionGraphUtils::GetProducerNodeName(input);
    const NodeDef* producer = FusionGraphUtils::GetNodeByName(graph, producer_name);
    
    if (producer && producer->op() == "RealDiv") {
      result.matched = true;
      result.captured_nodes["AddV2"] = node;
      result.captured_nodes["RealDiv"] = producer;
      return result;
    }
  }

  return FusionMatchResult{};
}

Status RealDivAddFusion::Apply(GraphDef* graph,
                               const FusionMatchResult& match_result) const {
  const NodeDef* add_node = match_result.captured_nodes.at("AddV2");
  const NodeDef* div_node = match_result.captured_nodes.at("RealDiv");

  VLOG(1) << "Applying RealDivAddFusion: fusing " << div_node->name() 
          << " and " << add_node->name() << " into MusafusedRealDivAdd";

  // Create Fused Node
  NodeDef* fused_node = graph->add_node();
  fused_node->set_op("MusafusedRealDivAdd");
  fused_node->set_device(add_node->device());

  // Input 0: RealDiv's Dividend (x)
  // Input 1: RealDiv's Divisor (y)
  // Input 2: AddV2's other input (z)
  
  fused_node->add_input(div_node->input(0));
  fused_node->add_input(div_node->input(1));
  
  // Find the input of AddV2 that is NOT the RealDiv node
  bool found_other_input = false;
  for (const std::string& input : add_node->input()) {
    if (FusionGraphUtils::GetProducerNodeName(input) != div_node->name()) {
      fused_node->add_input(input);
      found_other_input = true;
      break;
    }
  }
  
  // If Add has two inputs from the same RealDiv? (edge case, but Handle it)
  if (!found_other_input) {
      fused_node->add_input(div_node->name());
  }

  // Copy attributes
  if (add_node->attr().count("T")) {
    (*fused_node->mutable_attr())["T"] = add_node->attr().at("T");
  }

  // Update graph: Replace Add with Fused node
  std::string original_name = add_node->name();
  fused_node->set_name(original_name);
  
  // To avoid name collision during Apply, we temporarily rename the old add_node
  // But wait, GraphDef is a protobuf, we are adding a new node.
  // The standard way in this's plugin is to rename the original node.
  const_cast<NodeDef*>(add_node)->set_name(original_name + "/old");

  return Status::OK();
}

bool RealDivAddFusion::IsKernelAvailable() const {
    return true; 
}

REGISTER_FUSION_PATTERN(RealDivAddFusion);

REGISTER_FUSION_KERNEL(RealDivAddFusion, []() {
  return true;
});

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
