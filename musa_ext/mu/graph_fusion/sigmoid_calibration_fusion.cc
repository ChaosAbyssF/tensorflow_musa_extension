/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mu/graph_fusion/sigmoid_calibration_fusion.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

#include "mu/graph_fusion/fusion_pattern_manager.h"
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

  std::string node_name = FusionGraphUtils::GetProducerNodeName(input);

  for (int i = 0; i < graph.node_size(); ++i) {
    if (graph.node(i).name() == node_name) {
      return &graph.node(i);
    }
  }
  return nullptr;
}

// Helper to check if a const node has a specific float value
bool HasFloatValue(const NodeDef& node, float expected_val,
                   float tolerance = 1e-5f) {
  if (!IsOp(node, "Const")) return false;

  auto it = node.attr().find("value");
  if (it == node.attr().end() || !it->second.has_tensor()) {
    return false;
  }

  const auto& tensor = it->second.tensor();
  if (tensor.float_val_size() > 0) {
    return std::abs(tensor.float_val(0) - expected_val) < tolerance;
  }

  return false;
}

}  // namespace

// =============================================================================
// MusaSigmoidCalibrationFusion Implementation
// =============================================================================

MusaSigmoidCalibrationFusion::MusaSigmoidCalibrationFusion() = default;

bool MusaSigmoidCalibrationFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaSigmoidCalibrationFusion::Match(
    const GraphDef& graph, int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& real_div_node = graph.node(start_node_idx);
  if (!IsOp(real_div_node, "RealDiv")) {
    return FusionMatchResult{};
  }

  FusionMatchResult result;

  // RealDiv input 0: Sigmoid(x)
  // RealDiv input 1: AddV2
  const NodeDef* sigmoid_node = FindProducer(graph, real_div_node.input(0));
  const NodeDef* add_node = FindProducer(graph, real_div_node.input(1));

  if (!sigmoid_node || !add_node || !IsOp(*sigmoid_node, "Sigmoid") ||
      (!IsOp(*add_node, "AddV2") && !IsOp(*add_node, "Add"))) {
    return FusionMatchResult{};
  }

  // Add input 0: Sigmoid(x) (same as above)
  // Add input 1: Mul
  const NodeDef* sigmoid_in_add = FindProducer(graph, add_node->input(0));
  const NodeDef* mul_node = FindProducer(graph, add_node->input(1));

  // Some graphs might have the inputs swapped
  if (sigmoid_in_add != sigmoid_node) {
    mul_node = FindProducer(graph, add_node->input(0));
    sigmoid_in_add = FindProducer(graph, add_node->input(1));
  }

  if (sigmoid_in_add != sigmoid_node || !mul_node || !IsOp(*mul_node, "Mul")) {
    return FusionMatchResult{};
  }

  // Mul input 0: Sub(1-S)
  // Mul input 1: Const (1x32)
  const NodeDef* sub_node = FindProducer(graph, mul_node->input(0));
  const NodeDef* scale_const_node = FindProducer(graph, mul_node->input(1));

  if (sub_node && IsOp(*sub_node, "Const")) {
    // Swapped case
    scale_const_node = sub_node;
    sub_node = FindProducer(graph, mul_node->input(1));
  }

  if (!sub_node || !IsOp(*sub_node, "Sub") || !scale_const_node ||
      !IsOp(*scale_const_node, "Const")) {
    return FusionMatchResult{};
  }

  // Sub input 0: Const (1)
  // Sub input 1: Sigmoid(x) (same as above)
  const NodeDef* one_const_node = FindProducer(graph, sub_node->input(0));
  const NodeDef* sigmoid_in_sub = FindProducer(graph, sub_node->input(1));

  if (!one_const_node || !sigmoid_in_sub || sigmoid_in_sub != sigmoid_node ||
      !HasFloatValue(*one_const_node, 1.0f)) {
    return FusionMatchResult{};
  }

  // Success!
  result.matched = true;
  result.matched_nodes = {&real_div_node, add_node, mul_node, sub_node,
                          sigmoid_node};

  result.captured_nodes["output"] = &real_div_node;
  result.captured_nodes["add"] = add_node;
  result.captured_nodes["mul"] = mul_node;
  result.captured_nodes["sub"] = sub_node;
  result.captured_nodes["sigmoid"] = sigmoid_node;

  result.captured_nodes["input"] = FindProducer(graph, sigmoid_node->input(0));
  result.captured_nodes["scale"] = scale_const_node;
  result.captured_nodes["one_const"] = one_const_node;

  return result;
}

Status MusaSigmoidCalibrationFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid FusedSigmoidCalibration match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  // Get captured nodes
  auto output_it = match_result.captured_nodes.find("output");
  auto sigmoid_it = match_result.captured_nodes.find("sigmoid");
  auto input_it = match_result.captured_nodes.find("input");
  auto scale_it = match_result.captured_nodes.find("scale");
  auto one_const_it = match_result.captured_nodes.find("one_const");

  if (output_it == match_result.captured_nodes.end() ||
      sigmoid_it == match_result.captured_nodes.end() ||
      input_it == match_result.captured_nodes.end() ||
      scale_it == match_result.captured_nodes.end() ||
      one_const_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required nodes in SigmoidCalibration pattern");
  }

  const NodeDef* output_node = output_it->second;
  const NodeDef* sigmoid_node = sigmoid_it->second;
  const NodeDef* input_node = input_it->second;
  const NodeDef* scale_const_node = scale_it->second;
  const NodeDef* one_const_node = one_const_it->second;

  const std::string original_name = output_node->name();
  const std::string original_output_name = original_name + "_original";

  // Check if this output node has already been fused (avoid duplicates)
  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaSigmoidCalibration") {
      VLOG(1) << "MusaSigmoidCalibration: Output node " << original_name
              << " is already a fused node, skipping";
      return Status::OK();
    }
  }

  int output_node_idx = FusionGraphUtils::FindNodeIndex(*graph, original_name);
  if (output_node_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to find output node in graph: " + original_name);
  }

  VLOG(INFO) << "MusaSigmoidCalibrationFusion: Replacing " << original_name
             << " with MusaSigmoidCalibration";

  NodeDef* original_node = graph->mutable_node(output_node_idx);
  const std::string output_device = original_node->device();

  DataType dtype = DT_FLOAT;
  auto it = original_node->attr().find("T");
  if (it != original_node->attr().end()) {
    dtype = it->second.type();
  }

  original_node->set_name(original_output_name);

  // 1. Add new fused node
  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaSigmoidCalibration");
  fused_node->set_device(output_device);

  fused_node->add_input(sigmoid_node->input(0));
  fused_node->add_input(scale_const_node->name());
  fused_node->add_input(one_const_node->name());

  (*fused_node->mutable_attr())["T"].set_type(dtype);

  VLOG(INFO) << "MusaSigmoidCalibration: Fused node added as " << original_name;

  // 2. Remove intermediate nodes if unused
  std::vector<std::string> removable_names = {
      original_output_name, match_result.captured_nodes.at("add")->name(),
      match_result.captured_nodes.at("mul")->name(),
      match_result.captured_nodes.at("sub")->name(), sigmoid_node->name()};

  FusionGraphUtils::RemoveNodesIfUnused(
      graph, removable_names,
      {input_node->name(), scale_const_node->name(), one_const_node->name(),
       original_name});

  // 检查融合算子确实被加入到图中了
  bool found_fused_node = false;
  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaSigmoidCalibration") {
      found_fused_node = true;
      LOG(INFO) << "MusaSigmoidCalibrationFusion: Successfully added fused node "
                << original_name << " to graph";
      break;
    }
  }
  if (!found_fused_node) {
    return Status(error::INTERNAL,
                  "Failed to add fused node to graph: " + original_name);
  }

  return Status::OK();
}

// 注册融合模式
REGISTER_FUSION_PATTERN(MusaSigmoidCalibrationFusion);

// 注册 kernel 可用性
REGISTER_FUSION_KERNEL(MusaSigmoidCalibrationFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
