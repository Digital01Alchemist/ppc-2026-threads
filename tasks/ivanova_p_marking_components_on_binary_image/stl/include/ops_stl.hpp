#pragma once

#include <vector>

#include "ivanova_p_marking_components_on_binary_image/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ivanova_p_marking_components_on_binary_image {

class IvanovaPMarkingComponentsOnBinaryImageSTL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSTL;
  }
  explicit IvanovaPMarkingComponentsOnBinaryImageSTL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  Image input_image_;
  std::vector<int> labels_;
  std::vector<int> parent_;
  int current_label_ = 0;
  int width_ = 0;
  int height_ = 0;

  int FindRoot(int label);
  void UnionLabels(int label1, int label2);
  void ProcessPixel(int xx, int yy, int idx);
  void FirstPass();
  void SecondPass();

  // Helper methods for strip-based processing with local DSU
  void ProcessStripPixel(int xx, int yy, int idx, int strip_start_row, std::vector<int> &local_parent,
                         int &local_label);
  void MergeStripBoundaries(int num_threads, int rows_per_thread);
  static int FindLocalRoot(int label, const std::vector<int> &local_parent);
  static void UnionLocalLabels(int label1, int label2, std::vector<int> &local_parent);
  static void InitializeLocalParent(std::vector<int> &local_parent, int max_labels);
  void ProcessStrip(int start_row, int end_row, std::vector<int> &local_parent, int &local_label);
  void MergeLocalParents(const std::vector<std::vector<int>> &local_parents, const std::vector<int> &local_labels,
                         int num_threads, int total_pixels);
};

}  // namespace ivanova_p_marking_components_on_binary_image