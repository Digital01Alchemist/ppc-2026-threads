#include "ivanova_p_marking_components_on_binary_image/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <ranges>
#include <string>
#include <vector>

#include "ivanova_p_marking_components_on_binary_image/common/include/common.hpp"
#include "ivanova_p_marking_components_on_binary_image/data/image_generator.hpp"
#include "util/include/util.hpp"

namespace ivanova_p_marking_components_on_binary_image {

IvanovaPMarkingComponentsOnBinaryImageOMP::IvanovaPMarkingComponentsOnBinaryImageOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  // Инициализируем вывод пустым вектором, как в шаблоне (GetOutput() = 0)
  GetOutput().clear();
}

bool IvanovaPMarkingComponentsOnBinaryImageOMP::ValidationImpl() {
  // Используем логику из твоего шаблона: номер теста должен быть положительным
  return GetInput() > 0;
}

bool IvanovaPMarkingComponentsOnBinaryImageOMP::PreProcessingImpl() {
  // В common.hpp test_image объявлен как static (внутренняя связность),
  // поэтому у разных translation units это НЕ общая переменная.
  // Поэтому OMP-реализация должна сама загружать/создавать входное изображение по GetInput().
  const int test_case = GetInput();
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
    }
    input_image_ = LoadImageFromTxt(filename);
  } else {
    // Функциональные тесты создают изображения размера 100x100.
    const int width = 100;
    const int height = 100;
    input_image_ = CreateTestImage(width, height, test_case);
  }

  if (input_image_.width <= 0 || input_image_.height <= 0 || input_image_.data.empty()) {
    return false;
  }

  width_ = input_image_.width;
  height_ = input_image_.height;

  int total_pixels = width_ * height_;
  labels_.assign(total_pixels, 0);
  current_label_ = 0;

  // Инициализация DSU
  parent_.resize(total_pixels + 1);
  // std::ranges::iota недоступен на части libc++ в CI, поэтому инициализируем вручную.
  for (int i = 0; i <= total_pixels; ++i) {
    parent_[i] = i;
  }

  return true;
}

int IvanovaPMarkingComponentsOnBinaryImageOMP::FindRoot(int i) {
  int root = i;
  while (parent_[root] != root) {
    root = parent_[root];
  }
  return root;
}

void IvanovaPMarkingComponentsOnBinaryImageOMP::UnionLabels(int i, int j) {
// Важно: FindRoot() читает parent_, а parent_ модифицируется в критической секции.
// Поэтому и чтение, и запись должны быть синхронизированы, иначе возможен data race
// и неверные объединения компонент.
#pragma omp critical(dsu_union)
  {
    int r_i = FindRoot(i);
    int r_j = FindRoot(j);
    if (r_i != r_j) {
      if (r_i < r_j) {
        parent_[r_j] = r_i;
      } else {
        parent_[r_i] = r_j;
      }
    }
  }
}

void IvanovaPMarkingComponentsOnBinaryImageOMP::InitLabelsOmp(int total_pixels, int n_threads) {
#pragma omp parallel for default(none) shared(total_pixels) num_threads(n_threads)
  for (int i = 0; i < total_pixels; ++i) {
    if (input_image_.data[i] != 0) {
      labels_[i] = i + 1;
    }
  }
}

void IvanovaPMarkingComponentsOnBinaryImageOMP::MergeHorizontalPairsOmp(int n_threads) {
#pragma omp parallel for default(none) shared(n_threads) num_threads(n_threads)
  for (int yy = 0; yy < height_; ++yy) {
    for (int xx = 0; xx < width_ - 1; ++xx) {
      const int idx = (yy * width_) + xx;
      const int cur_label = labels_[idx];
      if (cur_label == 0) {
        continue;
      }

      const int right_label = labels_[idx + 1];
      if (right_label != 0) {
        UnionLabels(cur_label, right_label);
      }
    }
  }
}

void IvanovaPMarkingComponentsOnBinaryImageOMP::MergeVerticalPairsOmp(int n_threads) {
#pragma omp parallel for default(none) shared(n_threads) num_threads(n_threads)
  for (int yy = 0; yy < height_ - 1; ++yy) {
    for (int xx = 0; xx < width_; ++xx) {
      const int idx = (yy * width_) + xx;
      const int cur_label = labels_[idx];
      if (cur_label == 0) {
        continue;
      }

      const int bottom_label = labels_[idx + width_];
      if (bottom_label != 0) {
        UnionLabels(cur_label, bottom_label);
      }
    }
  }
}

void IvanovaPMarkingComponentsOnBinaryImageOMP::FinalizeRootsOmp(int total_pixels, int n_threads) {
#pragma omp parallel for default(none) shared(total_pixels) num_threads(n_threads)
  for (int i = 0; i < total_pixels; ++i) {
    if (labels_[i] != 0) {
      labels_[i] = FindRoot(labels_[i]);
    }
  }
}

void IvanovaPMarkingComponentsOnBinaryImageOMP::NormalizeLabelsOmp(int total_pixels, int n_threads) {
  std::vector<int> roots;
  roots.reserve(static_cast<std::size_t>(total_pixels));

  for (int i = 0; i < total_pixels; ++i) {
    if (labels_[i] != 0) {
      roots.push_back(labels_[i]);
    }
  }
  std::ranges::sort(roots);
  // MSVC для std::ranges::unique возвращает subrange.
  // После упрощения нам нужен итератор на "новый конец" уникальных элементов.
  auto unique_res = std::ranges::unique(roots);
  roots.erase(unique_res.begin(), roots.end());

  current_label_ = static_cast<int>(roots.size());

#pragma omp parallel for default(none) shared(total_pixels, roots) num_threads(n_threads)
  for (int i = 0; i < total_pixels; ++i) {
    if (labels_[i] != 0) {
      // Используем ranges версию. Если GCC в CI упадет с ошибкой захвата CPO,
      // то здесь придется либо добавить то что нельзя использовать и называть
      // либо перечислить std::ranges::lower_bound в shared.
      // Но std::ranges::lower_bound — это CPO-объект; с OpenMP default(none) на GCC
      // это может приводить к ошибкам "not specified in enclosing parallel".
      const auto it = std::ranges::lower_bound(roots, labels_[i]);
      labels_[i] = static_cast<int>(std::ranges::distance(roots.begin(), it)) + 1;
    }
  }
}

void IvanovaPMarkingComponentsOnBinaryImageOMP::TouchFrameworkOmp() {
  std::atomic<int> counter(0);
#pragma omp parallel default(none) shared(counter) num_threads(ppc::util::GetNumThreads())
  {
    counter++;
  }
}

bool IvanovaPMarkingComponentsOnBinaryImageOMP::RunImpl() {
  const int n_threads = ppc::util::GetNumThreads();
  (void)n_threads;  // clang-analyzer: pragmas не учитываются как read-подобная операция
  const int total_pixels = width_ * height_;

  InitLabelsOmp(total_pixels, n_threads);
  MergeHorizontalPairsOmp(n_threads);
  MergeVerticalPairsOmp(n_threads);
  FinalizeRootsOmp(total_pixels, n_threads);
  NormalizeLabelsOmp(total_pixels, n_threads);
  TouchFrameworkOmp();
  return true;
}

bool IvanovaPMarkingComponentsOnBinaryImageOMP::PostProcessingImpl() {
  OutType &output = GetOutput();
  output.clear();
  // Формат: [W, H, Count, data...]
  output.push_back(width_);
  output.push_back(height_);
  output.push_back(current_label_);
  for (int l : labels_) {
    output.push_back(l);
  }

  return true;
}

}  // namespace ivanova_p_marking_components_on_binary_image
