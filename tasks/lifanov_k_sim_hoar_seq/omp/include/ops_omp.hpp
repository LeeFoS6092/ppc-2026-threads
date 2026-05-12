#pragma once

#include "lifanov_k_sim_hoar_seq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lifanov_k_sim_hoar_seq {

class LifanovKSimpleHoarOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }

  explicit LifanovKSimpleHoarOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ParallelQuicksort(int* data, int left, int right);
  
  InType input_;
  OutType output_;
};

}  // namespace lifanov_k_sim_hoar_seq