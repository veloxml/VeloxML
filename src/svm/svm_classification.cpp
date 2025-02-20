// svm/svm_classification.cpp

#include "svm/svm_classification.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cfloat>
#include <random>
#include <cblas.h> // OpenBLAS のヘッダ

// OpenMP
#include <omp.h>
// TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

//////////////////////////
// コンストラクタ／デストラクタ
//////////////////////////
SVMClassification::SVMClassification(
    double C,
    double tol,
    int max_passes,
    const std::string &kernel,
    bool gamma_scale,
    double gamma,
    double coef0,
    int degree,
    bool approx_kernel)
    : C_(C),
      tol_(tol),
      max_passes_(max_passes),
      kernel_(kernel),
      gamma_scale_(gamma_scale),
      // gamma_scale_ が false の場合は入力値を設定、true の場合は fit() 内で自動計算
      gamma_(gamma_scale ? 0.0 : gamma),
      coef0_(coef0),
      degree_(degree),
      approx_kernel_(approx_kernel),
      b_(0.0),
      n_train_(0),
      m_features_(0),
      rff_dim_(100), // 固定値（必要に応じてパラメータ化可）
      is_platt_calibrated_(false),
      is_initialized_(false)
{
  // 必要に応じて初期化
}

SVMClassification::~SVMClassification()
{
  // リソース解放が必要な場合
}

//////////////////////////
// 内部：BLAS/OpenBLAS を用いた dot 積計算
//////////////////////////
double SVMClassification::dot(const double *a, const double *b, std::size_t len) const
{
  return cblas_ddot(static_cast<int>(len), a, 1, b, 1);
}

//////////////////////////
// 内部：サンプルインデックス i, j に対するカーネル計算
//////////////////////////
double SVMClassification::kernel_function_index(std::size_t i, std::size_t j) const
{
  // linear と poly では既存の cblas_ddot を利用
  if (kernel_ == "linear" || kernel_ == "poly")
  {
    double dp = cblas_ddot(static_cast<int>(m_features_),
                           &X_train_[i], static_cast<int>(n_train_),
                           &X_train_[j], static_cast<int>(n_train_));
    if (kernel_ == "linear")
      return dp;
    else // poly
      return std::pow(dp + coef0_, degree_);
  }
  else if (kernel_ == "rbf")
  {
    // cblas_ddot で内積を計算
    double dp = cblas_ddot(static_cast<int>(m_features_),
                           &X_train_[i], static_cast<int>(n_train_),
                           &X_train_[j], static_cast<int>(n_train_));
    // 事前に計算したノルムを利用して ||x_i - x_j||^2 を計算
    double norm_sq = sample_norms_[i] + sample_norms_[j] - 2.0 * dp;
    return std::exp(-gamma_ * norm_sq);
  }
  else
  {
    throw std::invalid_argument("Unsupported kernel type");
  }
}

//////////////////////////
// 内部：連続メモリ上のベクトルに対するカーネル計算
//////////////////////////
double SVMClassification::kernel_function(const double *x1, const double *x2) const
{
  double dp = cblas_ddot(static_cast<int>(m_features_), x1, 1, x2, 1);
  if (kernel_ == "linear")
  {
    return dp;
  }
  else if (kernel_ == "rbf")
  {
    double norm_sq = 0.0;
    for (std::size_t i = 0; i < m_features_; ++i)
    {
      double d = x1[i] - x2[i];
      norm_sq += d * d;
    }
    return std::exp(-gamma_ * norm_sq);
  }
  else if (kernel_ == "poly")
  {
    return std::pow(dp + coef0_, degree_);
  }
  else
  {
    throw std::invalid_argument("Unsupported kernel type");
  }
}

//////////////////////////
// 内部：近似カーネル法（Random Fourier Features）の実装
//////////////////////////
void SVMClassification::compute_approx_features(const std::vector<double> &X,
                                                std::vector<double> &X_approx,
                                                std::size_t l, std::size_t m) const
{
  // X, X_approx はともに ColMajor 形式
  X_approx.resize(l * rff_dim_);
  double scale = std::sqrt(2.0 / rff_dim_);
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, l),
                    [&](const tbb::blocked_range<std::size_t> &r)
                    {
                      std::vector<double> sample(m);
                      std::vector<double> transformed(rff_dim_);
                      for (std::size_t i = r.begin(); i != r.end(); ++i)
                      {
                        for (std::size_t j = 0; j < m; ++j)
                        {
                          sample[j] = X[j * l + i];
                        }
                        for (std::size_t d = 0; d < rff_dim_; ++d)
                        {
                          double ip = cblas_ddot(static_cast<int>(m), &rff_weights_[d * m], 1, sample.data(), 1);
                          transformed[d] = std::cos(ip + rff_bias_[d]) * scale;
                        }
                        for (std::size_t d = 0; d < rff_dim_; ++d)
                        {
                          X_approx[d * l + i] = transformed[d];
                        }
                      }
                    });
}

//////////////////////////
// 学習関数：SMO 最適化 & Platt scaling キャリブレーション（エラーキャッシュ、働く集合選択、シュリンキング、カーネルキャッシュ付き）
//////////////////////////
void SVMClassification::fit(const std::vector<double> &X,
                            const std::vector<double> &Y,
                            std::size_t n, std::size_t m)
{
  // 学習データを保存（ColMajor 形式）
  n_train_ = n;
  m_features_ = m;
  X_train_ = X;
  Y_train_ = Y;

  is_initialized_ = true;

  // gamma_scale_ が true の場合、学習データに基づいて gamma を計算する
  if (gamma_scale_)
  {
    double sum = 0.0, sum_sq = 0.0;
    std::size_t total = n_train_ * m_features_;
    for (std::size_t i = 0; i < total; ++i)
    {
      double val = X_train_[i];
      sum += val;
      sum_sq += val * val;
    }
    double mean = sum / total;
    double variance = sum_sq / total - mean * mean;
    if (variance <= 0)
    {
      variance = 1.0; // 0除算回避
    }
    // scikit-learn の "scale" と同様の定義: gamma = 1 / (n_features * variance)
    gamma_ = 1.0 / (m_features_ * variance);
  }

  // 近似カーネル法を利用する場合、Random Fourier Features の初期化と入力変換
  if (approx_kernel_)
  {
    // 乱数ジェネレータの初期化（再現性のため固定シード）
    std::mt19937 gen(42);
    std::normal_distribution<> gauss(0.0, 1.0);
    std::uniform_real_distribution<> uni(0.0, 2 * M_PI);
    // rff_weights_ のサイズは (rff_dim_ x m_features_)、rff_bias_ のサイズは (rff_dim_)
    rff_weights_.resize(rff_dim_ * m_features_);
    rff_bias_.resize(rff_dim_);
    for (std::size_t i = 0; i < rff_dim_ * m_features_; ++i)
      rff_weights_[i] = gauss(gen) * std::sqrt(2 * gamma_);
    for (std::size_t i = 0; i < rff_dim_; ++i)
      rff_bias_[i] = uni(gen);
    // 入力データを変換し、特徴空間次元を rff_dim_ に更新する
    std::vector<double> X_approx;
    compute_approx_features(X_train_, X_approx, n_train_, m_features_);
    X_train_ = X_approx;
    m_features_ = rff_dim_;
    // 近似カーネル利用時は内部的に線形SVMとして扱うため、カーネルタイプは "linear" と同様に扱う
    // kernel_ = "linear";
  }

  // カーネルキャッシュの利用：線形の場合は不要
  bool use_cache = false;
  if (kernel_ == "rbf" || kernel_ == "poly")
  {
    use_cache = true;
    kernel_cache_.resize(n_train_ * n_train_);
    if (kernel_ == "rbf")
    {
      // RBF用：各学習サンプルの二乗ノルムを事前計算
      sample_norms_.resize(n_train_);
      for (std::size_t i = 0; i < n_train_; ++i)
      {
        double norm_sq = 0.0;
        for (std::size_t k = 0; k < m_features_; ++k)
        {
          double x = X_train_[k * n_train_ + i];
          norm_sq += x * x;
        }
        sample_norms_[i] = norm_sq;
      }
      // カーネル行列の計算（対称行列として計算）
      for (std::size_t i = 0; i < n_train_; ++i)
      {
        for (std::size_t j = i; j < n_train_; ++j)
        {
          double dp = cblas_ddot(static_cast<int>(m_features_),
                                 &X_train_[i], static_cast<int>(n_train_),
                                 &X_train_[j], static_cast<int>(n_train_));
          double norm_sq = sample_norms_[i] + sample_norms_[j] - 2.0 * dp;
          double k_val = std::exp(-gamma_ * norm_sq);
          kernel_cache_[i * n_train_ + j] = k_val;
          kernel_cache_[j * n_train_ + i] = k_val;
        }
      }
    }
    else if (kernel_ == "poly")
    {
      for (std::size_t i = 0; i < n_train_; ++i)
      {
        for (std::size_t j = i; j < n_train_; ++j)
        {
          double dp = cblas_ddot(static_cast<int>(m_features_),
                                 &X_train_[i], static_cast<int>(n_train_),
                                 &X_train_[j], static_cast<int>(n_train_));
          double k_val = std::pow(dp + coef0_, degree_);
          kernel_cache_[i * n_train_ + j] = k_val;
          kernel_cache_[j * n_train_ + i] = k_val;
        }
      }
    }
  }

  // SMO 初期化
  alphas_.assign(n_train_, 0.0);
  b_ = 0.0;
  errors_.assign(n_train_, 0.0);
  for (std::size_t i = 0; i < n_train_; ++i)
  {
    errors_[i] = -Y_train_[i]; // 初期 f(x)=b=0 より
  }
  // 活動中サンプルの初期化（全サンプルを活性化）
  active_set_.assign(n_train_, true);

  int passes = 0;
  const double eps = 1e-5;
  const int max_passes_modified = std::max(max_passes_, 100);
  std::mt19937 rng(42);

  while (passes < max_passes_modified)
  {
    int num_changed_alphas = 0;
    for (std::size_t i = 0; i < n_train_; ++i)
    {
      if (!active_set_[i])
        continue; // 非活性サンプルはスキップ

      double f_i = 0.0;
      if (use_cache)
      {
        for (std::size_t j = 0; j < n_train_; ++j)
        {
          f_i += alphas_[j] * Y_train_[j] * kernel_cache_[j * n_train_ + i];
        }
      }
      else
      {
        for (std::size_t j = 0; j < n_train_; ++j)
        {
          f_i += alphas_[j] * Y_train_[j] * kernel_function_index(j, i);
        }
      }
      f_i += b_;
      double E_i = f_i - Y_train_[i];

      // KKT 条件違反チェック
      if ((Y_train_[i] * E_i < -tol_ && alphas_[i] < C_) ||
          (Y_train_[i] * E_i > tol_ && alphas_[i] > 0))
      {

        // 第2変数 j の選択：活性なサンプルの中から、|E_i - E_j| が最大のものを選択
        double maxDelta = 0.0;
        std::size_t j = 0;
        bool found = false;
        for (std::size_t k = 0; k < n_train_; ++k)
        {
          if (!active_set_[k] || k == i)
            continue;
          double delta = std::fabs(E_i - errors_[k]);
          if (delta > maxDelta)
          {
            maxDelta = delta;
            j = k;
            found = true;
          }
        }
        if (!found)
          j = (i + 1) % n_train_;

        double f_j = 0.0;
        if (use_cache)
        {
          for (std::size_t k = 0; k < n_train_; ++k)
          {
            f_j += alphas_[k] * Y_train_[k] * kernel_cache_[k * n_train_ + j];
          }
        }
        else
        {
          for (std::size_t k = 0; k < n_train_; ++k)
          {
            f_j += alphas_[k] * Y_train_[k] * kernel_function_index(k, j);
          }
        }
        f_j += b_;
        double E_j = f_j - Y_train_[j];

        double alpha_i_old = alphas_[i];
        double alpha_j_old = alphas_[j];

        double L, H;
        if (Y_train_[i] != Y_train_[j])
        {
          L = std::max(0.0, alphas_[j] - alphas_[i]);
          H = std::min(C_, C_ + alphas_[j] - alphas_[i]);
        }
        else
        {
          L = std::max(0.0, alphas_[i] + alphas_[j] - C_);
          H = std::min(C_, alphas_[i] + alphas_[j]);
        }
        if (std::fabs(L - H) < eps)
          continue;

        double Kii, Kij, Kjj;
        if (use_cache)
        {
          Kii = kernel_cache_[i * n_train_ + i];
          Kij = kernel_cache_[i * n_train_ + j];
          Kjj = kernel_cache_[j * n_train_ + j];
        }
        else
        {
          Kii = kernel_function_index(i, i);
          Kij = kernel_function_index(i, j);
          Kjj = kernel_function_index(j, j);
        }
        double eta = 2 * Kij - Kii - Kjj;
        if (eta >= 0)
          continue;

        alphas_[j] = alphas_[j] - (Y_train_[j] * (E_i - E_j)) / eta;
        alphas_[j] = std::min(H, std::max(L, alphas_[j]));
        if (std::fabs(alphas_[j] - alpha_j_old) < eps)
          continue;

        alphas_[i] = alphas_[i] + Y_train_[i] * Y_train_[j] * (alpha_j_old - alphas_[j]);

        double b1 = b_ - E_i - Y_train_[i] * (alphas_[i] - alpha_i_old) * Kii -
                    Y_train_[j] * (alphas_[j] - alpha_j_old) * Kij;
        double b2 = b_ - E_j - Y_train_[i] * (alphas_[i] - alpha_i_old) * Kij -
                    Y_train_[j] * (alphas_[j] - alpha_j_old) * Kjj;
        if (alphas_[i] > 0 && alphas_[i] < C_)
          b_ = b1;
        else if (alphas_[j] > 0 && alphas_[j] < C_)
          b_ = b2;
        else
          b_ = (b1 + b2) / 2.0;

        // エラーキャッシュの更新（i, j のみ再計算）
        double new_f_i = 0.0, new_f_j = 0.0;
        if (use_cache)
        {
          for (std::size_t k = 0; k < n_train_; ++k)
          {
            new_f_i += alphas_[k] * Y_train_[k] * kernel_cache_[k * n_train_ + i];
            new_f_j += alphas_[k] * Y_train_[k] * kernel_cache_[k * n_train_ + j];
          }
        }
        else
        {
          for (std::size_t k = 0; k < n_train_; ++k)
          {
            new_f_i += alphas_[k] * Y_train_[k] * kernel_function_index(k, i);
            new_f_j += alphas_[k] * Y_train_[k] * kernel_function_index(k, j);
          }
        }
        new_f_i += b_;
        new_f_j += b_;
        errors_[i] = new_f_i - Y_train_[i];
        errors_[j] = new_f_j - Y_train_[j];

        num_changed_alphas++;
      }

      // シュリンキング：各活性サンプルについて、KKT条件を十分満たしている場合は非活性化
      for (std::size_t i = 0; i < n_train_; ++i)
      {
        if (active_set_[i])
        {
          double f_i = 0.0;
          if (use_cache)
          {
            for (std::size_t j = 0; j < n_train_; ++j)
              f_i += alphas_[j] * Y_train_[j] * kernel_cache_[j * n_train_ + i];
          }
          else
          {
            for (std::size_t j = 0; j < n_train_; ++j)
              f_i += alphas_[j] * Y_train_[j] * kernel_function_index(j, i);
          }
          f_i += b_;
          // ここでは、ある程度マージンを持ってKKTを満たしていれば非活性化
          if ((Y_train_[i] * f_i >= 1 - tol_) || (Y_train_[i] * f_i <= -1 + tol_))
            active_set_[i] = false;
        }
      }

      if (num_changed_alphas == 0)
      {
        passes++;
        // 10パスごとに全サンプルを再活性化（シュリンキングの再調整）
        if (passes % 10 == 0)
        {
          std::fill(active_set_.begin(), active_set_.end(), true);
        }
      }
      else
      {
        passes = 0;
      }
    }

    // 線形または近似カーネルの場合、重みベクトル w_ を BLAS で計算
    if (kernel_ == "linear" || approx_kernel_)
    {
      std::vector<double> z(n_train_);
      for (std::size_t i = 0; i < n_train_; ++i)
        z[i] = alphas_[i] * Y_train_[i];
      w_.resize(m_features_);
      cblas_dgemv(CblasColMajor, CblasTrans,
                  static_cast<int>(n_train_), static_cast<int>(m_features_),
                  1.0, X_train_.data(), static_cast<int>(n_train_),
                  z.data(), 1, 0.0, w_.data(), 1);
    }

    // Platt scaling によるキャリブレーション
    std::vector<double> train_scores = predict_score(X_train_, n_train_, m_features_);
    std::vector<double> platt_targets(n_train_);
    for (std::size_t i = 0; i < n_train_; ++i)
      platt_targets[i] = (Y_train_[i] > 0) ? 1.0 : 0.0;
    platt_model_.fit(train_scores, platt_targets, n_train_, 1);
    is_platt_calibrated_ = true;
  }
}

//////////////////////////
// 予測：ラベル出力（決定関数の符号で分類）
//////////////////////////
std::vector<double> SVMClassification::predict(
    const std::vector<double> &X,
    std::size_t l, std::size_t m)
{
  std::vector<double> scores = predict_score(X, l, m);
  std::vector<double> preds(l, 0.0);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < l; ++i)
  {
    preds[i] = (scores[i] >= 0) ? 1.0 : -1.0;
  }
  return preds;
}

//////////////////////////
// スコア予測：各サンプルごとに決定関数値を返す
//////////////////////////
std::vector<double> SVMClassification::predict_score(const std::vector<double> &X,
                                                     std::size_t l, std::size_t m)
{
  std::vector<double> scores(l, 0.0);
  std::vector<double> X_input;
  std::size_t m_input = m;

  // 近似カーネルの場合は入力を変換
  if (approx_kernel_)
  {
    compute_approx_features(X, X_input, l, m);
    m_input = rff_dim_;
  }
  else
  {
    X_input = X;
  }

  // 線形または近似カーネルの場合は BLAS による一括計算
  if (kernel_ == "linear" || approx_kernel_)
  {
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                static_cast<int>(l), static_cast<int>(m_input),
                1.0, X_input.data(), static_cast<int>(l),
                w_.data(), 1, 0.0, scores.data(), 1);
    for (std::size_t i = 0; i < l; ++i)
    {
      scores[i] += b_;
    }
    return scores;
  }

  // 非線形カーネルの場合 ("rbf" や "poly")
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, l),
                    [&](const tbb::blocked_range<std::size_t> &r)
                    {
                      // 各スレッドごとに作業用バッファを1度だけ確保
                      std::vector<double> sample(m_input);
                      // 直接 X_train_ からアクセスすることで余分なコピーを避ける
                      for (std::size_t i = r.begin(); i != r.end(); ++i)
                      {
                        for (std::size_t j = 0; j < m_input; ++j)
                        {
                          sample[j] = X_input[j * l + i];
                        }
                        double sum = 0.0;
                        for (std::size_t k = 0; k < n_train_; ++k)
                        {
                          double kernel_val = 0.0;
                          if (kernel_ == "rbf")
                          {
                            double norm_sq = 0.0;
                            for (std::size_t j = 0; j < m_input; ++j)
                            {
                              double diff = sample[j] - X_train_[j * n_train_ + k];
                              norm_sq += diff * diff;
                            }
                            kernel_val = std::exp(-gamma_ * norm_sq);
                          }
                          else if (kernel_ == "poly")
                          {
                            double dp = 0.0;
                            for (std::size_t j = 0; j < m_input; ++j)
                            {
                              dp += sample[j] * X_train_[j * n_train_ + k];
                            }
                            kernel_val = std::pow(dp + coef0_, degree_);
                          }
                          sum += alphas_[k] * Y_train_[k] * kernel_val;
                        }
                        scores[i] = sum + b_;
                      }
                    });
  return scores;
}

//////////////////////////
// 確率推定：Platt scaling によるキャリブレーションを適用
//////////////////////////
std::vector<double> SVMClassification::predict_proba(const std::vector<double> &X,
                                                     std::size_t l, std::size_t m)
{
  std::vector<double> scores = predict_score(X, l, m);
  std::vector<double> prob(l * 2, 0.0); // [p(-1), p(+1)]
  if (is_platt_calibrated_)
  {
    std::vector<double> calibrated = platt_model_.predict_proba(scores, l, 1);
    for (std::size_t i = 0; i < l; ++i)
    {
      double p = calibrated[i];
      prob[i * 2 + 0] = 1.0 - p;
      prob[i * 2 + 1] = p;
    }
  }
  else
  {
    for (std::size_t i = 0; i < l; ++i)
    {
      double p = 1.0 / (1.0 + std::exp(-scores[i]));
      prob[i * 2 + 0] = 1 - p;
      prob[i * 2 + 1] = p;
    }
  }
  return prob;
}

//////////////////////////
// ゲッター群
//////////////////////////
double SVMClassification::getC() const { return C_; }
double SVMClassification::getTol() const { return tol_; }
int SVMClassification::getMaxPasses() const { return max_passes_; }
std::string SVMClassification::getKernel() const { return kernel_; }
double SVMClassification::getGamma() const { return gamma_; }
double SVMClassification::getCoef0() const { return coef0_; }
int SVMClassification::getDegree() const { return degree_; }
bool SVMClassification::getApproxKernel() const { return approx_kernel_; }
