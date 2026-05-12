#include "lifanov_k_sim_hoar_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include "lifanov_k_sim_hoar_seq/common/include/common.hpp"
#include "util/include/util.hpp"

namespace lifanov_k_sim_hoar_seq {

LifanovKSimpleHoarOMP::LifanovKSimpleHoarOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool LifanovKSimpleHoarOMP::ValidationImpl() {
  return true;
}

bool LifanovKSimpleHoarOMP::PreProcessingImpl() {
  output_ = input_;
  return true;
}

bool LifanovKSimpleHoarOMP::RunImpl() {
  if (output_.empty()) return true;

  const int num_threads = ppc::util::GetNumThreads();

  #pragma omp parallel num_threads(num_threads) default(none) shared(output_)
  {
    #pragma omp single nowait
    {
      ParallelQuicksort(output_.data(), 0, static_cast<int>(output_.size()) - 1);
    }
  }

  return true;
}

bool LifanovKSimpleHoarOMP::PostProcessingImpl() {
  GetOutput() = output_;
  return true;
}

void LifanovKSimpleHoarOMP::ParallelQuicksort(int *data, int left, int right) {
  if (left >= right) return;

  int i = left;
  int j = right;
  int pivot = data[left + ((right - left) / 2)];

  while (i <= j) {
    while (data[i] < pivot) i++;
    while (data[j] > pivot) j--;
    if (i <= j) {
      std::swap(data[i], data[j]);
      i++;
      j--;
    }
  }

  if (right - left > 4096) {
    #pragma omp task default(none) shared(data) firstprivate(left, j)
    ParallelQuicksort(data, left, j);

    #pragma omp task default(none) shared(data) firstprivate(i, right)
    ParallelQuicksort(data, i, right);
    
    #pragma omp taskwait
  } else {
    if (left < j) ParallelQuicksort(data, left, j);
    if (i < right) ParallelQuicksort(data, i, right);
  }
}

}  // namespace lifanov_k_sim_hoar_seq