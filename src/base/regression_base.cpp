#include "base/regression_base.hpp"

// 🔥 これにより、仮想関数テーブル（vtable）が確実に生成される
RegressionBase::~RegressionBase() = default;
