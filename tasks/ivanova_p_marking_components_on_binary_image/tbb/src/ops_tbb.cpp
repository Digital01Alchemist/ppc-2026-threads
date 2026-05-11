#include "ivanova_p_marking_components_on_binary_image/tbb/include/ops_tbb.hpp"
#include "ivanova_p_marking_components_on_binary_image/data/image_generator.hpp"
#include <tbb/tbb.h>
#include <algorithm>
#include <string>
#include <vector>

namespace ivanova_p_marking_components_on_binary_image {

IvanovaPMarkingComponentsOnBinaryImageTBB::IvanovaPMarkingComponentsOnBinaryImageTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
  
  // КРИТИЧНО: Сбрасываем глобальное изображение для чистоты тестов
  test_image.width = 0;
  test_image.height = 0;
  test_image.data.clear();
}

bool IvanovaPMarkingComponentsOnBinaryImageTBB::ValidationImpl() {
  if (test_image.width <= 0 || test_image.height <= 0) {
    int test_case = GetInput();
    if (test_case >= 11 && test_case <= 14) {
      std::string filename;
      switch (test_case) {
        case 11: filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image.txt"; break;
        case 12: filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image2.txt"; break;
        case 13: filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image3.txt"; break;
        case 14: filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image4.txt"; break;
      }
      test_image = LoadImageFromTxt(filename);
    } else {
      test_image = CreateTestImage(100, 100, test_case);
    }
  }
  return test_image.width > 0 && !test_image.data.empty();
}

bool IvanovaPMarkingComponentsOnBinaryImageTBB::PreProcessingImpl() {
  input_image_ = test_image;
  width_ = input_image_.width;
  height_ = input_image_.height;
  int total_pixels = width_ * height_;

  labels_.assign(total_pixels, 0);
  parent_.resize(total_pixels + 1);
  for (int i = 0; i <= total_pixels; ++i) parent_[i] = i;

  current_label_ = 0;
  return true;
}

int IvanovaPMarkingComponentsOnBinaryImageTBB::FindRoot(int i) {
  int root = i;
  while (parent_[root] != root) {
    root = parent_[root];
  }
  return root;
}

void IvanovaPMarkingComponentsOnBinaryImageTBB::UnionLabels(int i, int j) {
  tbb::spin_mutex::scoped_lock lock(dsu_mutex_);
  int root_i = FindRoot(i);
  int root_j = FindRoot(j);
  if (root_i != root_j) {
    if (root_i < root_j) parent_[root_j] = root_i;
    else parent_[root_i] = root_j;
  }
}

void IvanovaPMarkingComponentsOnBinaryImageTBB::InitLabelsTbb(int total_pixels) {
  tbb::parallel_for(0, total_pixels, [this](int i) {
    if (input_image_.data[i] != 0) {
      labels_[i] = i + 1;
    }
  });
}

void IvanovaPMarkingComponentsOnBinaryImageTBB::MergeHorizontalPairsTbb() {
  tbb::parallel_for(0, height_, [this](int yy) {
    for (int xx = 0; xx < width_ - 1; ++xx) {
      int idx = yy * width_ + xx;
      if (labels_[idx] != 0 && labels_[idx + 1] != 0) {
        UnionLabels(labels_[idx], labels_[idx + 1]);
      }
    }
  });
}

void IvanovaPMarkingComponentsOnBinaryImageTBB::MergeVerticalPairsTbb() {
  tbb::parallel_for(0, height_ - 1, [this](int yy) {
    for (int xx = 0; xx < width_; ++xx) {
      int idx = yy * width_ + xx;
      if (labels_[idx] != 0 && labels_[idx + width_] != 0) {
        UnionLabels(labels_[idx], labels_[idx + width_]);
      }
    }
  });
}

void IvanovaPMarkingComponentsOnBinaryImageTBB::FinalizeRootsTbb(int total_pixels) {
  tbb::parallel_for(0, total_pixels, [this](int i) {
    if (labels_[i] != 0) {
      labels_[i] = FindRoot(labels_[i]);
    }
  });
}

void IvanovaPMarkingComponentsOnBinaryImageTBB::NormalizeLabelsTbb(int total_pixels) {
  // Последовательная нормализация для гарантии правильного порядка меток (1, 2, 3...)
  std::vector<int> mapping(total_pixels + 1, 0);
  int next_id = 1;
  for (int i = 0; i < total_pixels; ++i) {
    if (labels_[i] != 0) {
      int root = labels_[i];
      if (mapping[root] == 0) {
        mapping[root] = next_id++;
      }
      labels_[i] = mapping[root];
    }
  }
  current_label_ = next_id - 1;
}

bool IvanovaPMarkingComponentsOnBinaryImageTBB::RunImpl() {
  int total_pixels = width_ * height_;
  if (total_pixels <= 0) return true;

  InitLabelsTbb(total_pixels);
  MergeHorizontalPairsTbb();
  MergeVerticalPairsTbb();
  FinalizeRootsTbb(total_pixels);
  NormalizeLabelsTbb(total_pixels);

  return true;
}

bool IvanovaPMarkingComponentsOnBinaryImageTBB::PostProcessingImpl() {
  OutType &output = GetOutput();
  output.clear();
  output.reserve(3 + labels_.size());
  output.push_back(width_);
  output.push_back(height_);
  output.push_back(current_label_);
  for (int l : labels_) output.push_back(l);
  return true;
}

}  // namespace ivanova_p_marking_components_on_binary_image