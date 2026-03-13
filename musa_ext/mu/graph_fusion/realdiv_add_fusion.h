#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_REALDIV_ADD_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_REALDIV_ADD_FUSION_H_

#include <string>
#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

class RealDivAddFusion : public FusionPattern {
 public:
  RealDivAddFusion() = default;
  ~RealDivAddFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph, int start_node_idx) const override;
  Status Apply(GraphDef* graph, const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 100; }
  bool IsKernelAvailable() const override;
  std::string GetName() const override { return "RealDivAddFusion"; }
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_REALDIV_ADD_FUSION_H_
