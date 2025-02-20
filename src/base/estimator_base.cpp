#include "base/estimator_base.hpp"

// 🔥 これにより、仮想関数テーブル（vtable）が確実に生成される
EstimatorBase::~EstimatorBase() = default;
