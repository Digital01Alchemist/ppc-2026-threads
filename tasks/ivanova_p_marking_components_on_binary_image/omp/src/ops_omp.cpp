#include "ivanova_p_marking_components_on_binary_image/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <numeric>
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
  std::iota(parent_.begin(), parent_.end(), 0);

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

bool IvanovaPMarkingComponentsOnBinaryImageOMP::RunImpl() {
  const int n_threads = ppc::util::GetNumThreads();
  const int total_pixels = width_ * height_;

// 1. Начальная разметка (параллельно)
#pragma omp parallel for default(none) shared(total_pixels) num_threads(n_threads)
  for (int i = 0; i < total_pixels; ++i) {
    if (input_image_.data[i] != 0) {
      labels_[i] = i + 1;
    }
  }

// 2. Слияние соседей (параллельно)
#pragma omp parallel for default(none) shared(n_threads) num_threads(n_threads)
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int idx = y * width_ + x;
      if (labels_[idx] == 0) {
        continue;
      }

      if (x + 1 < width_ && labels_[idx + 1] != 0) {
        UnionLabels(labels_[idx], labels_[idx + 1]);
      }
      if (y + 1 < height_ && labels_[idx + width_] != 0) {
        UnionLabels(labels_[idx], labels_[idx + width_]);
      }
    }
  }

// 3. Финализация корней
#pragma omp parallel for default(none) shared(total_pixels) num_threads(n_threads)
  for (int i = 0; i < total_pixels; ++i) {
    if (labels_[i] != 0) {
      labels_[i] = FindRoot(labels_[i]);
    }
  }

  // 4. Нормализация меток (нужно для прохождения CheckTestOutputData)
  std::vector<int> roots;
  for (int i = 0; i < total_pixels; ++i) {
    if (labels_[i] != 0) {
      roots.push_back(labels_[i]);
    }
  }
  std::sort(roots.begin(), roots.end());
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());

  current_label_ = static_cast<int>(roots.size());

#pragma omp parallel for default(none) shared(total_pixels, roots) num_threads(n_threads)
  for (int i = 0; i < total_pixels; ++i) {
    if (labels_[i] != 0) {
      auto it = std::lower_bound(roots.begin(), roots.end(), labels_[i]);
      labels_[i] = static_cast<int>(std::distance(roots.begin(), it)) + 1;
    }
  }

  // Вставка «важной строки» из твоего шаблона для корректной работы с фреймворком
  std::atomic<int> counter(0);
#pragma omp parallel default(none) shared(counter) num_threads(ppc::util::GetNumThreads())
  {
    counter++;
  }

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
