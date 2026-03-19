#ifdef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_PER_TOKEN_FFN_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_PER_TOKEN_FFN_FUSION_H

#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {
class MusaPerTokenFFNFusion : public FusionPattern {
 public:
  MusaPerTokenFFNFusion() = default;
  ~MusaPerTokenFFNFusion() override = default;

  // Match the PerTokenFFN pattern starting from a node
  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;

  // Apply the fusion: replace matched subgraph with MusaPerTokenFFN
  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  // Priority: typical for FFN fusions
  int GetPriority() const override { return 100; }

  // Check if MusaPerTokenFFN kernel is available
  bool IsKernelAvailable() const override;

  std::string GetName() const override { return "MusaPerTokenFFNFusion"; }

 private:
  mutable bool kernel_available_ = false;
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_PER_TOKEN_FFN_FUSION_H