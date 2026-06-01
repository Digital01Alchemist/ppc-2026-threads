#include "ivanova_p_marking_components_on_binary_image/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include "ivanova_p_marking_components_on_binary_image/common/include/common.hpp"
#include "ivanova_p_marking_components_on_binary_image/data/image_generator.hpp"
#include "util/include/util.hpp"

namespace ivanova_p_marking_components_on_binary_image {

IvanovaPMarkingComponentsOnBinaryImageSTL::IvanovaPMarkingComponentsOnBinaryImageSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();

  test_image.width = 0;
  test_image.height = 0;
  test_image.data.clear();
}

bool IvanovaPMarkingComponentsOnBinaryImageSTL::ValidationImpl() {
  if (test_image.width <= 0 || test_image.height <= 0) {
    int test_case = GetInput();

    if (test_case >= 11 && test_case <= 14) {
      std::string filename;
      switch (test_case) {
        case 11:
          filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image.txt";
          break;
        case 12:
          filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image2.txt";
          break;
        case 13:
          filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image3.txt";
          break;
        case 14:
          filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image4.txt";
          break;
        default:
          filename = "";
          break;
      }
      test_image = LoadImageFromTxt(filename);
    } else {
      int size = ExtractImageSize(test_case);
      test_image = CreateTestImage(size, size, test_case);
    }
  }

  if (test_image.width <= 0 || test_image.height <= 0 || test_image.data.empty()) {
    return false;
  }
  if (test_image.data.size() != static_cast<size_t>(test_image.width) * static_cast<size_t>(test_image.height)) {
    return false;
  }
  return true;
}

bool IvanovaPMarkingComponentsOnBinaryImageSTL::PreProcessingImpl() {
  if (test_image.width <= 0 || test_image.height <= 0) {
    int test_case = GetInput();

    if (test_case >= 11 && test_case <= 14) {
      std::string filename;
      switch (test_case) {
        case 11:
          filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image.txt";
          break;
        case 12:
          filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image2.txt";
          break;
        case 13:
          filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image3.txt";
          break;
        case 14:
          filename = "tasks/ivanova_p_marking_components_on_binary_image/data/image4.txt";
          break;
        default:
          filename = "";
          break;
      }
      test_image = LoadImageFromTxt(filename);
    } else {
      int size = ExtractImageSize(test_case);
      test_image = CreateTestImage(size, size, test_case);
    }
  }

  input_image_ = test_image;
  width_ = input_image_.width;
  height_ = input_image_.height;

  int total_pixels = width_ * height_;
  labels_.assign(total_pixels, 0);

  // Initialize DSU with size +1 for labels starting from 1
  parent_.resize(total_pixels + 1);
  for (int i = 0; i <= total_pixels; ++i) {
    parent_[i] = i;
  }

  current_label_ = 0;
  return true;
}

int IvanovaPMarkingComponentsOnBinaryImageSTL::FindRoot(int label) {
  int root = label;
  while (parent_[static_cast<size_t>(root)] != root) {
    root = parent_[static_cast<size_t>(root)];
  }

  // Path compression
  while (parent_[static_cast<size_t>(label)] != label) {
    int next = parent_[static_cast<size_t>(label)];
    parent_[static_cast<size_t>(label)] = root;
    label = next;
  }
  return root;
}

void IvanovaPMarkingComponentsOnBinaryImageSTL::UnionLabels(int label1, int label2) {
  int root1 = FindRoot(label1);
  int root2 = FindRoot(label2);

  if (root1 != root2) {
    if (root1 < root2) {
      parent_[static_cast<size_t>(root2)] = root1;
    } else {
      parent_[static_cast<size_t>(root1)] = root2;
    }
  }
}

void IvanovaPMarkingComponentsOnBinaryImageSTL::ProcessStripPixel(int xx, int yy, int idx, int strip_start_row) {
  if (input_image_.data[static_cast<size_t>(idx)] == 0) {
    return;
  }

  int left_label = (xx > 0) ? labels_[static_cast<size_t>(idx - 1)] : 0;
  int top_label = (yy > strip_start_row) ? labels_[static_cast<size_t>(idx - width_)] : 0;

  if (left_label == 0 && top_label == 0) {
    labels_[static_cast<size_t>(idx)] = idx + 1;
  } else if (left_label != 0 && top_label == 0) {
    labels_[static_cast<size_t>(idx)] = left_label;
  } else if (left_label == 0 && top_label != 0) {
    labels_[static_cast<size_t>(idx)] = top_label;
  } else {
    // Both neighbors exist
    if (left_label == top_label) {
      labels_[static_cast<size_t>(idx)] = left_label;
    } else {
      labels_[static_cast<size_t>(idx)] = left_label;
      // Inline union for better performance
      int root1 = left_label;
      while (parent_[static_cast<size_t>(root1)] != root1) {
        root1 = parent_[static_cast<size_t>(root1)];
      }
      int root2 = top_label;
      while (parent_[static_cast<size_t>(root2)] != root2) {
        root2 = parent_[static_cast<size_t>(root2)];
      }
      if (root1 != root2) {
        if (root1 < root2) {
          parent_[static_cast<size_t>(root2)] = root1;
        } else {
          parent_[static_cast<size_t>(root1)] = root2;
        }
      }
    }
  }
}

void IvanovaPMarkingComponentsOnBinaryImageSTL::MergeStripBoundaries(int num_threads, int rows_per_thread) {
  for (int thread_id = 0; thread_id < num_threads - 1; ++thread_id) {
    int boundary_row = (thread_id + 1) * rows_per_thread;
    if (boundary_row >= height_) {
      continue;
    }

    for (int xx = 0; xx < width_; ++xx) {
      int top_idx = ((boundary_row - 1) * width_) + xx;
      int bottom_idx = (boundary_row * width_) + xx;

      int top_label = labels_[static_cast<size_t>(top_idx)];
      int bottom_label = labels_[static_cast<size_t>(bottom_idx)];

      if (top_label != 0 && bottom_label != 0 && top_label != bottom_label) {
        UnionLabels(top_label, bottom_label);
      }
    }
  }
}

void IvanovaPMarkingComponentsOnBinaryImageSTL::FirstPass() {
  int num_threads = ppc::util::GetNumThreads();
  int rows_per_thread = (height_ + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(num_threads));

  // Phase 1: Parallel strip processing
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.emplace_back([this, thread_id, rows_per_thread]() {
      int start_row = thread_id * rows_per_thread;
      int end_row = std::min(start_row + rows_per_thread, height_);
      if (start_row >= height_) {
        return;
      }

      for (int yy = start_row; yy < end_row; ++yy) {
        for (int xx = 0; xx < width_; ++xx) {
          int idx = (yy * width_) + xx;
          ProcessStripPixel(xx, yy, idx, start_row);
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  // Phase 2: Sequential boundary merging
  MergeStripBoundaries(num_threads, rows_per_thread);

  current_label_ = 1;
}

void IvanovaPMarkingComponentsOnBinaryImageSTL::SecondPass() {
  int total_pixels = width_ * height_;
  int num_threads = ppc::util::GetNumThreads();
  int chunk_size = (total_pixels + num_threads - 1) / num_threads;

  // Phase 1: Parallel path compression
  {
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(num_threads));
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      threads.emplace_back([this, thread_id, chunk_size, total_pixels]() {
        int start = thread_id * chunk_size;
        int end = std::min(start + chunk_size, total_pixels);

        for (int i = start; i < end; ++i) {
          if (labels_[static_cast<size_t>(i)] != 0) {
            int root = labels_[static_cast<size_t>(i)];
            while (parent_[static_cast<size_t>(root)] != root) {
              root = parent_[static_cast<size_t>(root)];
            }
            labels_[static_cast<size_t>(i)] = root;
          }
        }
      });
    }

    for (auto &thread : threads) {
      thread.join();
    }
  }

  // Phase 2: Build label mapping (sequential - fast enough)
  std::vector<int> label_map(static_cast<size_t>(total_pixels) + 1, 0);
  int next_label = 1;
  for (int i = 0; i < total_pixels; ++i) {
    if (labels_[static_cast<size_t>(i)] != 0) {
      int root = labels_[static_cast<size_t>(i)];
      if (label_map[static_cast<size_t>(root)] == 0) {
        label_map[static_cast<size_t>(root)] = next_label++;
      }
    }
  }
  current_label_ = next_label - 1;

  // Phase 3: Parallel label remapping
  {
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(num_threads));
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      threads.emplace_back([this, thread_id, chunk_size, total_pixels, &label_map]() {
        int start = thread_id * chunk_size;
        int end = std::min(start + chunk_size, total_pixels);

        for (int i = start; i < end; ++i) {
          if (labels_[static_cast<size_t>(i)] != 0) {
            labels_[static_cast<size_t>(i)] = label_map[static_cast<size_t>(labels_[static_cast<size_t>(i)])];
          }
        }
      });
    }

    for (auto &thread : threads) {
      thread.join();
    }
  }
}

bool IvanovaPMarkingComponentsOnBinaryImageSTL::RunImpl() {
  int total_pixels = width_ * height_;
  if (total_pixels <= 0) {
    return false;
  }

  FirstPass();

  if (current_label_ > 0) {
    SecondPass();
  }

  return true;
}

bool IvanovaPMarkingComponentsOnBinaryImageSTL::PostProcessingImpl() {
  OutType &output = GetOutput();
  output.clear();

  output.push_back(width_);
  output.push_back(height_);
  output.push_back(current_label_);

  for (int label : labels_) {
    output.push_back(label);
  }

  return true;
}

}  // namespace ivanova_p_marking_components_on_binary_image
