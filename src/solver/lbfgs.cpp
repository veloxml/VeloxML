#include "solver/lbfgs.hpp"
#include <cmath>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <cblas.h>

// SIMD最適化用のインクルード（各命令セットに対応）
#ifdef __AVX512F__
#include <immintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ────────────────────────────────────────
//  内部：SIMDを用いたdot積の実装（条件コンパイル）
namespace
{
#ifdef __AVX512F__
  static inline double simd_dot_product_avx512(const double *a, const double *b, size_t n)
  {
    double sum = 0.0;
    __m512d vsum = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 7 < n; i += 8)
    {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      vsum = _mm512_add_pd(vsum, _mm512_mul_pd(va, vb));
    }
    double buf[8];
    _mm512_storeu_pd(buf, vsum);
    for (int j = 0; j < 8; ++j)
      sum += buf[j];
    for (; i < n; ++i)
      sum += a[i] * b[i];
    return sum;
  }
#endif

#ifdef __AVX2__
  static inline double simd_dot_product_avx2(const double *a, const double *b, size_t n)
  {
    double sum = 0.0;
    __m256d vsum = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < n; i += 4)
    {
      __m256d va = _mm256_loadu_pd(a + i);
      __m256d vb = _mm256_loadu_pd(b + i);
      vsum = _mm256_add_pd(vsum, _mm256_mul_pd(va, vb));
    }
    double buf[4];
    _mm256_storeu_pd(buf, vsum);
    for (int j = 0; j < 4; ++j)
      sum += buf[j];
    for (; i < n; ++i)
      sum += a[i] * b[i];
    return sum;
  }
#endif

#ifdef __ARM_NEON
  // ※ 一部プラットフォームではNEONによる倍精度演算がサポートされていないため、単純なループにフォールバック
  static inline double simd_dot_product_neon(const double *a, const double *b, size_t n)
  {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
      sum += a[i] * b[i];
    return sum;
  }
#endif
} // end anonymous namespace
// ────────────────────────────────────────

LBFGSSolver::LBFGSSolver(const Options &opts) : options_(opts) {}

double LBFGSSolver::dot_product(const std::vector<double> &a, const std::vector<double> &b) const
{
  std::size_t n = a.size();
#ifndef USE_BLAS
#if defined(__AVX512F__)
  return simd_dot_product_avx512(a.data(), b.data(), n);
#elif defined(__AVX2__)
  return simd_dot_product_avx2(a.data(), b.data(), n);
#elif defined(__ARM_NEON)
  return simd_dot_product_neon(a.data(), b.data(), n);
#else
  // TBBを用いた並列reduceによるフォールバック
  double result = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, n),
      0.0,
      [&](const tbb::blocked_range<size_t> &r, double init) -> double
      {
        for (size_t i = r.begin(); i != r.end(); ++i)
        {
          init += a[i] * b[i];
        }
        return init;
      },
      std::plus<double>());
  return result;
#endif
#else
  return cblas_ddot(n, a.data(), 1, b.data(), 1);
#endif
}

void LBFGSSolver::axpy(double alpha, const std::vector<double> &x, std::vector<double> &y) const
{
  std::size_t n = x.size();
  cblas_daxpy(n, alpha, x.data(), 1, y.data(), 1);
// #pragma omp parallel for
//   for (std::size_t i = 0; i < n; ++i)
//     y[i] += alpha * x[i];
}

void LBFGSSolver::scale_vector(std::vector<double> &v, double alpha) const
{
  std::size_t n = v.size();
  cblas_dscal(n, alpha, v.data(), 1);
// #pragma omp parallel for
//   for (std::size_t i = 0; i < n; ++i)
//     v[i] *= alpha;
}

double LBFGSSolver::norm(const std::vector<double> &v) const
{
  return cblas_dnrm2(v.size(), v.data(), 1);
  // double sum = tbb::parallel_reduce(
  //     tbb::blocked_range<size_t>(0, v.size()),
  //     0.0,
  //     [&](const tbb::blocked_range<size_t> &r, double init) -> double
  //     {
  //       for (size_t i = r.begin(); i != r.end(); ++i)
  //         init += v[i] * v[i];
  //       return init;
  //     },
  //     std::plus<double>());
  // return std::sqrt(sum);
}

// ────────────────────────────────────────
//  バックトラックラインサーチ（Armijo条件）
//  theta_candidate = theta + alpha * p として，
//  f(theta_candidate) <= f(theta) + c * alpha * (grad^T * p) となる alpha を探索
double LBFGSSolver::line_search(const std::vector<double> &theta,
                                const std::vector<double> &grad,
                                const std::vector<double> &p,
                                std::function<double(const std::vector<double> &, std::vector<double> &)> func,
                                double f,
                                double &new_f,
                                std::vector<double> &new_grad)
{
  double alpha = options_.line_search_alpha;
  std::vector<double> theta_candidate(theta.size());
  double grad_dot_p = dot_product(grad, p);
  while (true)
  {
    // theta_candidate = theta + alpha * p
    theta_candidate = theta;
#pragma omp parallel for
    for (std::size_t i = 0; i < theta.size(); ++i)
      theta_candidate[i] += alpha * p[i];

    new_f = func(theta_candidate, new_grad);
    // Armijo条件
    if (new_f <= f + options_.line_search_c * alpha * grad_dot_p)
      break;
    alpha *= options_.line_search_rho;
    if (alpha < 1e-10)
      break; // step size が極小
  }
  return alpha;
}
// ────────────────────────────────────────

double LBFGSSolver::solve(std::vector<double> &theta,
                          std::function<double(const std::vector<double> &, std::vector<double> &)> func)
{

  int n = static_cast<int>(theta.size());
  std::vector<double> grad(n, 0.0);
  double f = func(theta, grad);
  double grad_norm = norm(grad);
  int k = 0;

  // 履歴初期化
  s_history_.clear();
  y_history_.clear();
  rho_history_.clear();

  std::vector<double> q(n), r(n), p(n);

  while (grad_norm > options_.tolerance && k < options_.max_iterations)
  {
    // 2-loop recursion による探索方向計算
    q = grad; // q に勾配をコピー
    int history_size = static_cast<int>(s_history_.size());
    std::vector<double> alpha(history_size, 0.0);

    // 1ループ目（逆順）
    for (int i = history_size - 1; i >= 0; --i)
    {
      alpha[i] = rho_history_[i] * dot_product(s_history_[i], q);
      axpy(-alpha[i], y_history_[i], q); // q = q - alpha[i] * y[i]
    }

    // 初期Hessian近似： gamma = (s_last^T y_last) / (y_last^T y_last)
    double gamma = 1.0;
    if (history_size > 0)
    {
      const std::vector<double> &s_last = s_history_.back();
      const std::vector<double> &y_last = y_history_.back();
      double sy = dot_product(s_last, y_last);
      double yy = dot_product(y_last, y_last);
      if (yy > 0)
        gamma = sy / yy;
    }
    r = q;
    scale_vector(r, gamma);

    // 2ループ目（順方向）
    for (int i = 0; i < history_size; ++i)
    {
      double beta = rho_history_[i] * dot_product(y_history_[i], r);
      axpy(alpha[i] - beta, s_history_[i], r); // r = r + s[i]*(alpha[i]-beta)
    }

    // 探索方向 p = -r
    p = r;
    scale_vector(p, -1.0);

    // ラインサーチ
    std::vector<double> new_grad(n, 0.0);
    double new_f;
    double step = line_search(theta, grad, p, func, f, new_f, new_grad);

    // theta 更新： theta_new = theta + step * p
    std::vector<double> theta_new = theta;
#pragma omp parallel for
    for (std::size_t i = 0; i < theta.size(); ++i)
      theta_new[i] += step * p[i];

    // 履歴更新： s = theta_new - theta, y = new_grad - grad
    std::vector<double> s(n), y(n);
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
      s[i] = theta_new[i] - theta[i];
      y[i] = new_grad[i] - grad[i];
    }
    double ys = dot_product(y, s);
    if (ys > 1e-10)
    {
      if (static_cast<int>(s_history_.size()) == options_.m)
      {
        // 履歴サイズ上限に達したら最古のデータを削除
        s_history_.erase(s_history_.begin());
        y_history_.erase(y_history_.begin());
        rho_history_.erase(rho_history_.begin());
      }
      s_history_.push_back(s);
      y_history_.push_back(y);
      rho_history_.push_back(1.0 / ys);
    }

    // 次反復へ更新
    theta = theta_new;
    grad = new_grad;
    f = new_f;
    grad_norm = norm(grad);
    ++k;
  }
  return f;
}
