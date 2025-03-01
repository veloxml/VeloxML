#include "svm/svm_regression.hpp"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

// 補助関数：Yの標準化（平均0、標準偏差1）
static void standardize_Y(const std::vector<double> &Y,
                          std::vector<double> &Y_scaled, double &mean,
                          double &stddev) {
  int n = Y.size();
  mean = std::accumulate(Y.begin(), Y.end(), 0.0) / n;
  Y_scaled.resize(n);
  for (int i = 0; i < n; i++) {
    Y_scaled[i] = Y[i] - mean;
  }
  double sq_sum = 0.0;
  for (int i = 0; i < n; i++) {
    sq_sum += Y_scaled[i] * Y_scaled[i];
  }
  stddev = std::sqrt(sq_sum / n);
  if (stddev > 1e-12) {
    for (int i = 0; i < n; i++) {
      Y_scaled[i] /= stddev;
    }
  }
}

SVMRegression::SVMRegression(double C, double epsilon, double tol, int max_iter,
                             KernelType kernel_type, double gamma, int degree,
                             double coef0, int approx_dim)
    : C_(C), epsilon_(epsilon), tol_(tol), max_iter_(max_iter),
      kernel_type_(kernel_type), gamma_(gamma), degree_(degree), coef0_(coef0),
      approx_dim_(approx_dim), bias_(0.0), train_n_(0), train_m_(0),
      Y_mean_(0.0), Y_std_(1.0) {
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
}

SVMRegression::~SVMRegression() {}

// 内部カーネル計算：X_effはCol‐Major、m_effはその次元
double SVMRegression::kernel_function(const std::vector<double> &X_eff,
                                      std::size_t n, std::size_t m_eff,
                                      std::size_t i, std::size_t j) const {
  if (kernel_type_ == KernelType::LINEAR ||
      kernel_type_ == KernelType::APPROX_RBF) {
    return cblas_ddot((int)m_eff, &X_eff[i], (int)n, &X_eff[j], (int)n);
  } else if (kernel_type_ == KernelType::POLYNOMIAL) {
    double dot = 0.0;
    for (std::size_t k = 0; k < m_eff; k++) {
      dot += X_eff[k * n + i] * X_eff[k * n + j];
    }
    return std::pow(gamma_ * dot + coef0_, degree_);
  } else if (kernel_type_ == KernelType::RBF) {
    double sum = 0.0;
    for (std::size_t k = 0; k < m_eff; k++) {
      double diff = X_eff[k * n + i] - X_eff[k * n + j];
      sum += diff * diff;
    }
    return std::exp(-gamma_ * sum);
  }
  return 0.0;
}

// selectSecondIndex:
// TBBを利用して、サンプルiに対して最大の誤差差を示す候補jを探索
std::size_t
SVMRegression::selectSecondIndex(std::size_t i, const std::vector<double> &E,
                                 const std::vector<double> &alpha,
                                 const std::vector<double> &alpha_star) const {
  struct MaxDiff {
    double diff;
    std::size_t index;
    MaxDiff() : diff(0.0), index(std::numeric_limits<std::size_t>::max()) {}
  };

  auto combine = [](const MaxDiff &a, const MaxDiff &b) -> MaxDiff {
    return (a.diff > b.diff) ? a : b;
  };

  // 1. 非境界サンプルのみ対象
  auto func_nonbound = [i, &E, &alpha, &alpha_star,
                        this](const tbb::blocked_range<std::size_t> &r,
                              MaxDiff init) -> MaxDiff {
    for (std::size_t k = r.begin(); k != r.end(); ++k) {
      if (k == i)
        continue;
      // 非境界サンプルの定義：α[k]とα_star[k]が十分離れている
      if (alpha[k] > tol_ && alpha[k] < C_ - tol_ && alpha_star[k] > tol_ &&
          alpha_star[k] < C_ - tol_) {
        double diff = std::fabs(E[i] - E[k]);
        if (diff > init.diff) {
          init.diff = diff;
          init.index = k;
        }
      }
    }
    return init;
  };

  MaxDiff candidate =
      tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, E.size()),
                           MaxDiff(), func_nonbound, combine);

  // 2. もし非境界サンプルが見つからなければ全体から探す
  if (candidate.index == std::numeric_limits<std::size_t>::max()) {
    auto func_all = [i, &E](const tbb::blocked_range<std::size_t> &r,
                            MaxDiff init) -> MaxDiff {
      for (std::size_t k = r.begin(); k != r.end(); ++k) {
        if (k == i)
          continue;
        double diff = std::fabs(E[i] - E[k]);
        if (diff > init.diff) {
          init.diff = diff;
          init.index = k;
        }
      }
      return init;
    };
    candidate =
        tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, E.size()),
                             MaxDiff(), func_all, combine);
  }

  // 3. もし候補が依然として見つからなければ、ランダムに選ぶ
  if (candidate.index == std::numeric_limits<std::size_t>::max()) {
    std::default_random_engine rng(std::rand());
    std::uniform_int_distribution<std::size_t> dist(0, E.size() - 1);
    candidate.index = dist(rng);
  }
  return candidate.index;
}

// computeDualGap: 全サンプルの双対ギャップ（KKT違反の最大値）を計算
double
SVMRegression::computeDualGap(const std::vector<double> &X_eff, std::size_t n,
                              std::size_t m_eff,
                              const std::vector<double> &alpha,
                              const std::vector<double> &alpha_star) const {
  double gap = 0.0;
  for (std::size_t i = 0; i < n; i++) {
    double f_i = bias_;
    for (std::size_t j = 0; j < n; j++) {
      f_i +=
          (alpha[j] - alpha_star[j]) * kernel_function(X_eff, n, m_eff, j, i);
    }
    double E_i = f_i - Y_train_[i];
    double v1 = 0.0, v2 = 0.0;
    if (alpha[i] < C_)
      v1 = std::max(0.0, -(E_i + epsilon_));
    if (alpha[i] > 0)
      v2 = std::max(0.0, E_i + epsilon_);
    double gap_i = std::max(v1, v2);
    gap = std::max(gap, gap_i);
  }
  return gap;
}

// updatePair: i と j
// のペア更新。4通りのケースを網羅し、更新後のバイアスと誤差キャッシュも更新
bool SVMRegression::updatePair(std::size_t i, std::size_t j,
                               const std::vector<double> &X_eff, std::size_t n,
                               std::size_t m_eff, std::vector<double> &alpha,
                               std::vector<double> &alpha_star,
                               std::vector<double> &E) {
  // f_i, f_j の計算
  double f_i = bias_, f_j = bias_;
  for (std::size_t k = 0; k < n; k++) {
    double Kki = kernel_function(X_eff, n, m_eff, k, i);
    double Kkj = kernel_function(X_eff, n, m_eff, k, j);
    f_i += (alpha[k] - alpha_star[k]) * Kki;
    f_j += (alpha[k] - alpha_star[k]) * Kkj;
  }
  double E_i = f_i - Y_train_[i];
  double E_j = f_j - Y_train_[j];

  // 4ケースのKKT違反評価
  double viol1 = 0.0, viol2 = 0.0, viol3 = 0.0, viol4 = 0.0;
  bool case1 = false, case2 = false, case3 = false, case4 = false;
  // CASE 1: 更新対象: α[i] と α[j]
  if (((alpha[i] < C_ && E_i + epsilon_ < -tol_) ||
       (alpha[i] > 0 && E_i + epsilon_ > tol_)) &&
      ((alpha[j] < C_ && E_j + epsilon_ > tol_) ||
       (alpha[j] > 0 && E_j + epsilon_ < -tol_))) {
    case1 = true;
    viol1 = std::fabs(E_i + epsilon_) + std::fabs(E_j + epsilon_);
  }
  // CASE 2: 更新対象: α*[i] と α*[j]
  if (((alpha_star[i] < C_ && -E_i + epsilon_ < -tol_) ||
       (alpha_star[i] > 0 && -E_i + epsilon_ > tol_)) &&
      ((alpha_star[j] < C_ && -E_j + epsilon_ > tol_) ||
       (alpha_star[j] > 0 && -E_j + epsilon_ < -tol_))) {
    case2 = true;
    viol2 = std::fabs(-E_i + epsilon_) + std::fabs(-E_j + epsilon_);
  }
  // CASE 3: 更新対象: α[i] と α*[j]
  if (((alpha[i] < C_ && E_i + epsilon_ < -tol_) ||
       (alpha[i] > 0 && E_i + epsilon_ > tol_)) &&
      ((alpha_star[j] < C_ && -E_j + epsilon_ < -tol_) ||
       (alpha_star[j] > 0 && -E_j + epsilon_ > tol_))) {
    case3 = true;
    viol3 = std::fabs(E_i + epsilon_) + std::fabs(-E_j + epsilon_);
  }
  // CASE 4: 更新対象: α*[i] と α[j]
  if (((alpha_star[i] < C_ && -E_i + epsilon_ < -tol_) ||
       (alpha_star[i] > 0 && -E_i + epsilon_ > tol_)) &&
      ((alpha[j] < C_ && E_j + epsilon_ > tol_) ||
       (alpha[j] > 0 && E_j + epsilon_ < -tol_))) {
    case4 = true;
    viol4 = std::fabs(-E_i + epsilon_) + std::fabs(E_j + epsilon_);
  }

  // 最も大きな違反を持つケースを選ぶ
  int caseType = -1;
  double maxViol = 0.0;
  if (case1 && viol1 > maxViol) {
    maxViol = viol1;
    caseType = 1;
  }
  if (case2 && viol2 > maxViol) {
    maxViol = viol2;
    caseType = 2;
  }
  if (case3 && viol3 > maxViol) {
    maxViol = viol3;
    caseType = 3;
  }
  if (case4 && viol4 > maxViol) {
    maxViol = viol4;
    caseType = 4;
  }
  if (caseType == -1)
    return false;

  // 定義: θ[i] = α[i] - α*[i], θ[j] = α[j] - α*[j], s = θ[i] + θ[j] (固定)
  double theta_i_old = alpha[i] - alpha_star[i];
  double theta_j_old = alpha[j] - alpha_star[j];
  double s = theta_i_old + theta_j_old;

  double K_ii = kernel_function(X_eff, n, m_eff, i, i);
  double K_jj = kernel_function(X_eff, n, m_eff, j, j);
  double K_ij = kernel_function(X_eff, n, m_eff, i, j);
  double eta = K_ii + K_jj - 2 * K_ij;
  if (eta <= 1e-12)
    return false;

  // 更新量Δの計算
  double delta = (E_i - E_j) / eta;
  double new_theta_i, new_theta_j;
  if (caseType == 1 || caseType == 3) {
    new_theta_i = theta_i_old + delta;
    new_theta_j = theta_j_old - delta;
  } else {
    new_theta_i = theta_i_old - delta;
    new_theta_j = theta_j_old + delta;
  }

  // クリッピング：θは[-C_, C_]に収める。等式制約 s = new_theta_i +
  // new_theta_j固定
  double L = std::max(-C_, s - C_);
  double H = std::min(C_, s);
  new_theta_i = std::min(H, std::max(L, new_theta_i));
  new_theta_j = s - new_theta_i;

  // 再構成
  double new_alpha_i = (new_theta_i > 0) ? new_theta_i : 0.0;
  double new_alpha_star_i = (new_theta_i < 0) ? -new_theta_i : 0.0;
  double new_alpha_j = (new_theta_j > 0) ? new_theta_j : 0.0;
  double new_alpha_star_j = (new_theta_j < 0) ? -new_theta_j : 0.0;

  if (std::fabs(new_alpha_i - alpha[i]) < 1e-5 &&
      std::fabs(new_alpha_star_i - alpha_star[i]) < 1e-5 &&
      std::fabs(new_alpha_j - alpha[j]) < 1e-5 &&
      std::fabs(new_alpha_star_j - alpha_star[j]) < 1e-5)
    return false;

  double b1 = bias_ - E_i - (new_theta_i - theta_i_old) * K_ii;
  double b2 = bias_ - E_j - (new_theta_j - theta_j_old) * K_jj;
  double new_b = (b1 + b2) / 2.0;

  // 更新適用
  alpha[i] = new_alpha_i;
  alpha_star[i] = new_alpha_star_i;
  alpha[j] = new_alpha_j;
  alpha_star[j] = new_alpha_star_j;
  bias_ = new_b;

  // 改善された誤差キャッシュ更新：あらかじめK_i, K_jを計算
  std::vector<double> K_i(n), K_j(n);
#pragma omp parallel for
  for (std::ptrdiff_t k = 0; k < (std::ptrdiff_t)n; k++) {
    K_i[k] = kernel_function(X_eff, n, m_eff, i, k);
    K_j[k] = kernel_function(X_eff, n, m_eff, j, k);
  }
#pragma omp parallel for
  for (std::ptrdiff_t k = 0; k < (std::ptrdiff_t)n; k++) {
    E[k] += (new_theta_i - theta_i_old) * K_i[k] +
            (new_theta_j - theta_j_old) * K_j[k];
  }
  return true;
}

// fit関数：学習データを前処理し、SMOメインループで更新し、収束判定（双対ギャップ）を行う
void SVMRegression::fit(const std::vector<double> &X,
                        const std::vector<double> &Y, std::size_t n,
                        std::size_t m) {
  if (n == 0 || m == 0)
    throw std::invalid_argument("Invalid dimensions for X");
  if (X.size() != n * m || Y.size() != n)
    throw std::invalid_argument("Input size does not match dimensions");

  std::vector<double> Y_scaled;
  standardize_Y(Y, Y_scaled, Y_mean_, Y_std_);
  Y_train_ = Y_scaled;
  train_n_ = n;
  train_m_ = m;
  X_train_ = X;

  const std::vector<double> *X_eff_ptr = &X;
  std::size_t m_eff = m;
  std::vector<double> X_trans;
  if (kernel_type_ == KernelType::APPROX_RBF) {
    if (W_.empty()) {
      W_.resize(approx_dim_ * m);
      rff_bias_.resize(approx_dim_);
      for (int i = 0; i < approx_dim_; i++) {
        for (std::size_t j = 0; j < m; j++) {
          double u1 = (double)std::rand() / RAND_MAX;
          double u2 = (double)std::rand() / RAND_MAX;
          double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2 * M_PI * u2);
          W_[i * m + j] = z * std::sqrt(2.0 * gamma_);
        }
        rff_bias_[i] = ((double)std::rand() / RAND_MAX) * 2 * M_PI;
      }
    }
    X_trans.resize(approx_dim_ * n);
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)n; i++) {
      for (int k = 0; k < approx_dim_; k++) {
        double dot = 0.0;
        for (std::size_t j = 0; j < m; j++) {
          dot += W_[k * m + j] * X[j * n + i];
        }
        X_trans[k * n + i] =
            std::sqrt(2.0 / approx_dim_) * std::cos(dot + rff_bias_[k]);
      }
    }
    X_eff_ptr = &X_trans;
    m_eff = approx_dim_;
  }

  std::vector<double> alpha(n, 0.0), alpha_star(n, 0.0);
  bias_ = 0.0;
  std::vector<double> E(n);
  for (std::size_t i = 0; i < n; i++) {
    E[i] = -Y_scaled[i];
  }

  // tol_gap は収束判定用の絶対閾値（例: tol_gap = tol_ と同じ値を利用）
  double tol_gap = tol_;
  // relative_gap_threshold は相対変化率の閾値（例: 1e-4）
  double relative_gap_threshold = 1e-4;
  double prev_gap = std::numeric_limits<double>::max();

  int iter = 0;
  while (iter < max_iter_) {
    int num_changed = 0;
    for (std::size_t i = 0; i < n; i++) {
      // 候補探索：TBBを使ってサンプル i に対して最大違反のある j を選ぶ
      std::size_t j = selectSecondIndex(i, E, alpha, alpha_star);
      if (j != std::numeric_limits<std::size_t>::max()) {
        if (updatePair(i, j, *X_eff_ptr, n, m_eff, alpha, alpha_star, E))
          num_changed++;
      }
    }
    double gap = computeDualGap(*X_eff_ptr, n, m_eff, alpha, alpha_star);
    // 収束判定：更新がなくなったか、絶対ギャップが十分小さいか、または相対変化率が閾値以下の場合
    double relative_change = std::fabs(gap - prev_gap) / (prev_gap + 1e-12);
    if (num_changed == 0 || gap < tol_gap ||
        relative_change < relative_gap_threshold)
      break;
    iter++;
  }

  theta_dual_.resize(n);
  for (std::size_t i = 0; i < n; i++) {
    theta_dual_[i] = alpha[i] - alpha_star[i];
  }
  if (kernel_type_ == KernelType::LINEAR ||
      kernel_type_ == KernelType::APPROX_RBF) {
    weights_.assign(m_eff, 0.0);
    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t j = 0; j < m_eff; j++) {
        weights_[j] += (alpha[i] - alpha_star[i]) * (*X_eff_ptr)[j * n + i];
      }
    }
  }
}

// predict関数：テストデータXに対して、学習済みモデルを用い逆標準化した予測を返す
std::vector<double> SVMRegression::predict(const std::vector<double> &X,
                                           std::size_t n, std::size_t m) {
  if (m != train_m_)
    throw std::invalid_argument(
        "Test feature dimension does not match training dimension.");
  std::vector<double> predictions(n, 0.0);
  if (kernel_type_ == KernelType::LINEAR) {
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)n; i++) {
      double sum = bias_;
      for (std::size_t j = 0; j < m; j++) {
        sum += weights_[j] * X[j * n + i];
      }
      predictions[i] = sum * Y_std_ + Y_mean_;
    }
  } else if (kernel_type_ == KernelType::APPROX_RBF) {
    std::vector<double> X_trans(approx_dim_ * n);
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)n; i++) {
      for (int k = 0; k < approx_dim_; k++) {
        double dot = 0.0;
        for (std::size_t j = 0; j < m; j++) {
          dot += W_[k * m + j] * X[j * n + i];
        }
        X_trans[k * n + i] =
            std::sqrt(2.0 / approx_dim_) * std::cos(dot + rff_bias_[k]);
      }
    }
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)n; i++) {
      double sum = bias_;
      for (std::size_t k = 0; k < static_cast<std::size_t>(approx_dim_); k++) {
        sum += weights_[k] * X_trans[k * n + i];
      }
      predictions[i] = sum * Y_std_ + Y_mean_;
    }
  } else {
#pragma omp parallel for
    for (std::ptrdiff_t t = 0; t < (std::ptrdiff_t)n; t++) {
      double sum = bias_;
      for (std::size_t i = 0; i < train_n_; i++) {
        double k_val = 0.0;
        if (kernel_type_ == KernelType::POLYNOMIAL) {
          double dot = 0.0;
          for (std::size_t j = 0; j < m; j++) {
            dot += X_train_[j * train_n_ + i] * X[j * n + t];
          }
          k_val = std::pow(gamma_ * dot + coef0_, degree_);
        } else if (kernel_type_ == KernelType::RBF) {
          double sum_sq = 0.0;
          for (std::size_t j = 0; j < m; j++) {
            double diff = X_train_[j * train_n_ + i] - X[j * n + t];
            sum_sq += diff * diff;
          }
          k_val = std::exp(-gamma_ * sum_sq);
        }
        sum += theta_dual_[i] * k_val;
      }
      predictions[t] = sum * Y_std_ + Y_mean_;
    }
  }
  return predictions;
}

std::vector<double> SVMRegression::get_weights() const { return weights_; }
double SVMRegression::get_bias() const { return bias_; }
std::vector<double> SVMRegression::get_theta_dual() const {
  return theta_dual_;
}
