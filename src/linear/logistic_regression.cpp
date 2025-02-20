#include "linear/logistic_regression.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <cblas.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <tbb/parallel_reduce.h>
#include <vector>
#include <algorithm>

// コンストラクタ
LogisticRegressionL2Objective::LogisticRegressionL2Objective(const std::vector<double> &X_config, int n, int m,
                                                             const std::vector<double> &y,
                                                             double lambda)
    : X_(X_config), y_(y), lambda_(lambda), N_(static_cast<size_t>(n)), D_(static_cast<size_t>(m))
{
  if (X_.size() != N_ * D_)
  {
    throw std::runtime_error("X_config size does not match n * m");
  }
  if (y_.size() != N_)
  {
    throw std::runtime_error("y size does not match n");
  }
}

// operator() : 損失関数と勾配の計算
double LogisticRegressionL2Objective::operator()(const std::vector<double> &theta, std::vector<double> &grad)
{
  if (theta.size() != D_)
  {
    throw std::runtime_error("theta size does not match number of features");
  }

  std::vector<double> z(N_, 0.0);
  // z = X * theta, X: (N x D) Col-Major
  cblas_dgemv(CblasColMajor, CblasNoTrans, N_, D_,
              1.0, X_.data(), N_,
              theta.data(), 1,
              0.0, z.data(), 1);

  // p = 1/(1+exp(-z)), 並列化
  std::vector<double> p(N_, 0.0);
#pragma omp parallel for
  for (size_t i = 0; i < N_; i++)
  {
    double exp_neg = std::exp(-z[i]);
    p[i] = 1.0 / (1.0 + exp_neg);
  }

  // 損失計算: L = -sum( y*log(p) + (1-y)*log(1-p) ) + 0.5*lambda*||theta||^2
  const double eps = 1e-15;
  double loss = 0.0;
#pragma omp parallel for reduction(+ : loss)
  for (size_t i = 0; i < N_; i++)
  {
    loss += -y_[i] * std::log(p[i] + eps) - (1.0 - y_[i]) * std::log(1.0 - p[i] + eps);
  }
  double theta_norm = cblas_dnrm2(static_cast<int>(D_), theta.data(), 1);
  loss += 0.5 * lambda_ * theta_norm * theta_norm;

  // 勾配計算: grad = X^T*(p-y) + lambda*theta
  grad.assign(D_, 0.0);
  std::vector<double> diff(N_, 0.0);
#pragma omp parallel for
  for (size_t i = 0; i < N_; i++)
  {
    diff[i] = p[i] - y_[i];
  }
  // X^T * diff の計算
  cblas_dgemv(CblasColMajor, CblasTrans, N_, D_,
              1.0, X_.data(), N_,
              diff.data(), 1,
              0.0, grad.data(), 1);
// 正則化項
#pragma omp parallel for
  for (size_t j = 0; j < D_; j++)
  {
    grad[j] += lambda_ * theta[j];
  }

  return loss;
}

// gradient(): 単に operator() を呼び出す
std::vector<double> LogisticRegressionL2Objective::gradient(const std::vector<double> &theta)
{
  std::vector<double> grad;
  double dummy = (*this)(theta, grad);
  (void)dummy;
  return grad;
}

// hessian(): 従来の全 Hessian を計算（ここでは参考実装）
std::vector<std::vector<double>> LogisticRegressionL2Objective::hessian(const std::vector<double> &theta)
{
  if (theta.size() != D_)
  {
    throw std::runtime_error("theta size does not match number of features");
  }
  std::vector<double> z(N_, 0.0);
  cblas_dgemv(CblasColMajor, CblasNoTrans, N_, D_,
              1.0, X_.data(), N_,
              theta.data(), 1,
              0.0, z.data(), 1);
  std::vector<double> p(N_, 0.0);
#pragma omp parallel for
  for (size_t i = 0; i < N_; i++)
  {
    double exp_neg = std::exp(-z[i]);
    p[i] = 1.0 / (1.0 + exp_neg);
  }

  // Hessian: H = X^T * diag(p*(1-p)) * X + lambda * I
  std::vector<std::vector<double>> H(D_, std::vector<double>(D_, 0.0));
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < D_; i++)
  {
    for (size_t j = 0; j < D_; j++)
    {
      double sum = 0.0;
      for (size_t k = 0; k < N_; k++)
      {
        double w = p[k] * (1.0 - p[k]);
        sum += w * X_[i * N_ + k] * X_[j * N_ + k];
      }
      if (i == j)
      {
        sum += lambda_;
      }
      H[i][j] = sum;
    }
  }
  return H;
}

// hvp(): Hessian-vector 積を計算する関数
void LogisticRegressionL2Objective::hvp(const std::vector<double> &theta,
                                        const std::vector<double> &v,
                                        std::vector<double> &hv)
{
  if (theta.size() != D_ || v.size() != D_)
  {
    throw std::runtime_error("theta or v size does not match number of features");
  }
  // 1. z = X * theta
  std::vector<double> z(N_, 0.0);
  cblas_dgemv(CblasColMajor, CblasNoTrans, N_, D_,
              1.0, X_.data(), N_,
              theta.data(), 1,
              0.0, z.data(), 1);

  // 2. p = 1/(1+exp(-z))
  std::vector<double> p(N_, 0.0);
#pragma omp parallel for
  for (size_t i = 0; i < N_; i++)
  {
    double exp_neg = std::exp(-z[i]);
    p[i] = 1.0 / (1.0 + exp_neg);
  }

  // 3. w = X * v
  std::vector<double> w(N_, 0.0);
  cblas_dgemv(CblasColMajor, CblasNoTrans, N_, D_,
              1.0, X_.data(), N_,
              v.data(), 1,
              0.0, w.data(), 1);

  // 4. u = p*(1-p) .* w
  std::vector<double> u(N_, 0.0);
#pragma omp parallel for
  for (size_t i = 0; i < N_; i++)
  {
    u[i] = p[i] * (1.0 - p[i]) * w[i];
  }

  // 5. hv = X^T * u
  hv.assign(D_, 0.0);
  cblas_dgemv(CblasColMajor, CblasTrans, N_, D_,
              1.0, X_.data(), N_,
              u.data(), 1,
              0.0, hv.data(), 1);

// 6. hv += lambda * v
#pragma omp parallel for
  for (size_t j = 0; j < D_; j++)
  {
    hv[j] += lambda_ * v[j];
  }
}

// ──────────────────────────────
// コンストラクタ
LogisticRegression::LogisticRegression(
    LogisticRegressionSolverType solver,
    double lambda,
    double tol,
    int maxIter,
    double ls_alpha_init,
    double ls_rho,
    double ls_c,
    int history_size,
    int lbfgs_k)
    : solver_(solver),
      lambda_(lambda),
      tol_(tol),
      maxIter_(maxIter),
      ls_alpha_init_(ls_alpha_init),
      ls_rho_(ls_rho),
      ls_c_(ls_c),
      history_size_(history_size),
      lbfgs_k_(lbfgs_k),
      bias_(0.0)
{
}

// ──────────────────────────────
// fit（C++ API）
// X: n×m（col-major 1次元配列）、Y: n要素
void LogisticRegression::fit(const std::vector<double> &X,
                             const std::vector<double> &Y,
                             std::size_t n, std::size_t m)
{
  if (solver_ == LogisticRegressionSolverType::LBFGS)
  {
    fit_lbfgs(X, Y, n, m);
  }
  else if (solver_ == LogisticRegressionSolverType::NEWTON)
  {
    fit_newton(X, Y, n, m);
  }
  else if (solver_ == LogisticRegressionSolverType::CD)
  {
    fit_cd(X, Y, n, m);
  }
  else
  {
    throw std::runtime_error("Unknown solver type.");
  }
}

// ──────────────────────────────
// fit_lbfgsの内部実装
void LogisticRegression::fit_lbfgs(const std::vector<double> &X,
                                   const std::vector<double> &Y,
                                   std::size_t n, std::size_t m)
{
  // バイアス項を含めた特徴次元：D = m + 1
  std::size_t D = m + 1;
  std::vector<double> X_aug(n * D, 0.0);
// 既存のX（n×m）を前半のm列にコピー（Col-Major）
#pragma omp parallel for collapse(2)
  for (std::size_t j = 0; j < m; j++)
  {
    for (std::size_t i = 0; i < n; i++)
    {
      X_aug[j * n + i] = X[j * n + i];
    }
  }
// 最後の列（バイアス項）は全て1.0
#pragma omp parallel for
  for (std::size_t i = 0; i < n; i++)
  {
    X_aug[m * n + i] = 1.0;
  }

  // LogisticRegressionL2Objectiveは、引数としてX (col-major), n, D, Y, λを受け取る
  LogisticRegressionL2Objective objective(X_aug, n, D, Y, lambda_);

  auto func = [&objective](const std::vector<double> &theta, std::vector<double> &grad) -> double
  {
    return objective(theta, grad);
  };

  // Newton-CG用に、hessianを1次元（col-major）に変換するlambda
  auto hess_func = [&objective, D](const std::vector<double> &theta) -> std::vector<double>
  {
    std::vector<std::vector<double>> H2D = objective.hessian(theta);
    std::vector<double> H(D * D, 0.0);
    for (std::size_t i = 0; i < D; i++)
    {
      for (std::size_t j = 0; j < D; j++)
      {
        // H2Dは row-major と仮定し、col-majorに変換
        H[j * D + i] = H2D[i][j];
      }
    }
    return H;
  };

  // 初期パラメータtheta（重みとバイアスを含む）をゼロ初期化
  std::vector<double> theta(D, 0.0);

  double final_obj = 0.0;
  // LBFGSソルバーを利用
  LBFGSSolver::Options opts;
  opts.max_iterations = maxIter_;
  opts.tolerance = tol_;
  opts.m = history_size_;
  opts.line_search_alpha = ls_alpha_init_;
  opts.line_search_rho = ls_rho_;
  opts.line_search_c = ls_c_;
  LBFGSSolver solver(opts);
  final_obj = solver.solve(theta, func);
  // std::cout << "LBFGS final objective: " << final_obj << std::endl;

  // 重みはthetaの先頭m成分、バイアスはtheta[m]
  weights_.resize(m);
#pragma omp parallel for
  for (std::size_t j = 0; j < m; j++)
  {
    weights_[j] = theta[j];
  }
  bias_ = theta[m];
}

// ──────────────────────────────
// fit_newtonの内部実装（Newton-CGを利用）
void LogisticRegression::fit_newton(const std::vector<double> &X,
                                    const std::vector<double> &Y,
                                    std::size_t n, std::size_t m)
{
  // バイアス項を含めた特徴数: D = m + 1
  std::size_t D = m + 1;
  std::vector<double> X_aug(n * D, 0.0);

// X は Col-Major の n×m 行列なので、X_aug の先頭 m 列にコピー
#pragma omp parallel for collapse(2)
  for (std::size_t j = 0; j < m; j++)
  {
    for (std::size_t i = 0; i < n; i++)
    {
      X_aug[j * n + i] = X[j * n + i];
    }
  }
// 最後の列にバイアス項（全て 1.0）を設定
#pragma omp parallel for
  for (std::size_t i = 0; i < n; i++)
  {
    X_aug[m * n + i] = 1.0;
  }

  // LogisticRegressionL2Objective のインスタンスを作成
  // ※ コンストラクタは (X_aug, n, D, Y, lambda_) の順
  LogisticRegressionL2Objective objective(X_aug, static_cast<int>(n), static_cast<int>(D), Y, lambda_);

  // 目的関数とその勾配を計算する関数（C++ lambda）
  auto func = [&objective](const std::vector<double> &theta, std::vector<double> &grad) -> double
  {
    return objective(theta, grad);
  };

  // マトリックスフリー版 Hessian-vector 積の計算関数
  auto hvp_func = [&objective](const std::vector<double> &theta, const std::vector<double> &v, std::vector<double> &hv)
  {
    objective.hvp(theta, v, hv);
  };

  // 初期パラメータ theta をゼロ初期化（サイズ: D）
  std::vector<double> theta(D, 0.0);

  // Newton-CG ソルバーのオプションを設定（必要なハイパーパラメータはコンストラクタから保持）
  NewtonCGSolver::Options opts;
  opts.max_iterations = maxIter_;
  opts.tolerance = tol_;
  opts.cg_max_iterations = 50;    // 内側 CG の最大反復回数
  opts.cg_tolerance = tol_;       // 内側 CG の最低許容絶対誤差
  opts.cg_tolerance_factor = 0.1; // 内側 CG の相対誤差： tol_cg = 0.1 * ||grad||
  opts.line_search_alpha = ls_alpha_init_;
  opts.line_search_rho = ls_rho_;
  opts.line_search_c = ls_c_;
  opts.use_preconditioning = false; // 必要に応じて true にする

  // 改良版マトリックスフリー Newton-CG ソルバーを呼び出す
  NewtonCGSolver solver(opts);
  double final_obj = solver.solve(theta, func, hvp_func);
  // std::cout << "Newton-CG final objective: " << final_obj << std::endl;

  // 学習後、theta の先頭 m 成分が重み、最後の成分がバイアスとなる
  weights_.resize(m);
#pragma omp parallel for
  for (std::size_t j = 0; j < m; j++)
  {
    weights_[j] = theta[j];
  }
  bias_ = theta[m];
}

// ──────────────────────────────
// fit_cdは未実装の場合、LBFGSにフォールバック
void LogisticRegression::fit_cd(const std::vector<double> &X,
                                const std::vector<double> &Y,
                                std::size_t n, std::size_t m)
{
  fit_lbfgs(X, Y, n, m);
}

// ──────────────────────────────
// predict（C++ API）
// X: l×m（col-major）、各サンプルごとに z = dot(w, x)+bias を計算し、z>=0なら1, else 0
std::vector<double> LogisticRegression::predict(const std::vector<double> &X,
                                                std::size_t l, std::size_t m)
{
  std::vector<double> predictions(l, 0.0);
#pragma omp parallel for
  for (std::size_t i = 0; i < l; i++)
  {
    double z = bias_;
    for (std::size_t j = 0; j < m; j++)
    {
      z += weights_[j] * X[j * l + i];
    }
    predictions[i] = (z >= 0) ? 1.0 : 0.0;
  }
  return predictions;
}

// ──────────────────────────────
// predict_proba（C++ API）
// 各サンプルごとに確率 p = 1/(1+exp(-z)) を計算
std::vector<double> LogisticRegression::predict_proba(const std::vector<double> &X,
                                                      std::size_t l, std::size_t m)
{
  // 出力は 2 クラス分：1列目がクラス0, 2列目がクラス1
  std::vector<double> probabilities(2 * l, 0.0);

// 各サンプルごとに、z = bias_ + dot(weights_, x_i) を計算
// X は Col-Major なので、i 番目サンプルの特徴は X[j*l + i] (j=0,...,m-1)
#pragma omp parallel for
  for (std::size_t i = 0; i < l; i++)
  {
    // BLAS を用いて内積を計算：weights_ と &X[i] の組み合わせで stride=l（Col-Major）
    double dot = cblas_ddot(static_cast<int>(m), weights_.data(), 1, &X[i], static_cast<int>(l));
    double z = bias_ + dot;
    // exp() のベクトル化（SIMD最適化は環境依存のライブラリ等を利用するか自前実装）
    double p = 1.0 / (1.0 + std::exp(-z));
    // Col-Major 出力：最初の l 要素がクラス0、残りがクラス1
    probabilities[i] = 1.0 - p; // クラス0 の確率
    probabilities[i + l] = p;   // クラス1 の確率
  }
  return probabilities;
}

// ──────────────────────────────
// Getter実装
std::vector<double> LogisticRegression::get_weights() const { return weights_; }
double LogisticRegression::get_bias() const { return bias_; }
LogisticRegressionSolverType LogisticRegression::get_solver() const { return solver_; }
double LogisticRegression::get_lambda() const { return lambda_; }
double LogisticRegression::get_tol() const { return tol_; }
int LogisticRegression::get_maxIter() const { return maxIter_; }
double LogisticRegression::get_ls_alpha_init() const { return ls_alpha_init_; }
double LogisticRegression::get_ls_rho() const { return ls_rho_; }
double LogisticRegression::get_ls_c() const { return ls_c_; }
int LogisticRegression::get_history_size() const { return history_size_; }
int LogisticRegression::get_lbfgs_k() const { return lbfgs_k_; }
