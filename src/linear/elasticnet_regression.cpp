#include "linear/elasticnet_regression.hpp"
#include <cstring>
#include <cblas.h>
#include <lapacke.h>
// #include <omp.h>
#ifdef __AVX512F__
#include <immintrin.h> // AVX512 用
#elif defined(__AVX2__)
#include <immintrin.h> // AVX2
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

// ── コンストラクタ ─────────────────────────────────────────────
ElasticnetRegression::ElasticnetRegression(double lambda1, double lambda2, int max_iter, double tol,
                                           ElasticNetSolverMode mode, double admm_rho, bool penalize_bias)
    : lambda1_(lambda1), lambda2_(lambda2), max_iter_(max_iter), tol_(tol),
      solver_mode_(mode), admm_rho_(admm_rho), penalize_bias_(penalize_bias), bias_(0.0)
{
    if (max_iter_ <= 0)
        throw std::invalid_argument("max_iter must be > 0");
    if (tol_ <= 0)
        throw std::invalid_argument("tol must be > 0");
}

// ── 入力行列にバイアス項を付加 ─────────────────────────────
// X: (n×m) の Col‐Major 配列 → X_aug: (n×(m+1))（各列が連続）
std::vector<double> ElasticnetRegression::augmentWithBias(const double *X, int n, int m)
{
    int d = m + 1;
    std::vector<double> X_aug(n * d, 0.0);
    // 元の m 列を TBB で並列コピー
    tbb::parallel_for(0, m, [=, &X_aug](int j)
                      { std::memcpy(&X_aug[j * n], &X[j * n], n * sizeof(double)); });
// バイアス列（最後の列）を1.0で埋める（OpenMP で並列化）
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        X_aug[m * n + i] = 1.0;
    }
    return X_aug;
}

// ── Lipschitz 定数の計算 ─────────────────────────────
// X_aug^T X_aug を計算し、対角に lambda2_ を加えた行列の最大固有値を求める
double ElasticnetRegression::computeLipschitzConstant(const std::vector<double> &X_aug, int n, int d)
{
    std::vector<double> A(d * d, 0.0);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                d, d, n,
                1.0,
                X_aug.data(), n,
                X_aug.data(), n,
                0.0,
                A.data(), d);
    // 対角成分に lambda2_ を加える
    for (int i = 0; i < d; i++)
    {
        A[i * d + i] += lambda2_;
    }
    std::vector<double> eigvals(d, 0.0);
    std::vector<double> A_copy = A;
    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'U', d, A_copy.data(), d, eigvals.data());
    if (info != 0)
        throw std::runtime_error("Eigenvalue computation failed in computeLipschitzConstant");
    return *std::max_element(eigvals.begin(), eigvals.end());
}

// ── ソフト閾値演算（proximal operator） ─────────────────────────────
void ElasticnetRegression::softThreshold(const std::vector<double> &x, std::vector<double> &out, double threshold, int d)
{
    out.resize(x.size());
#pragma omp parallel for
    for (int i = 0; i < d; i++)
    {
        double val = x[i];
        if (!penalize_bias_ && i == d - 1)
        {
            out[i] = val;
        }
        else
        {
            if (val > threshold)
                out[i] = val - threshold;
            else if (val < -threshold)
                out[i] = val + threshold;
            else
                out[i] = 0.0;
        }
    }
}

// ── FISTA ソルバー ─────────────────────────────
void ElasticnetRegression::solveFISTA(const double *X, int n, int m, const double *Y)
{
    int d = m + 1;
    std::vector<double> X_aug = augmentWithBias(X, n, m); // (n×d)
    double L_const = computeLipschitzConstant(X_aug, n, d);

    std::vector<double> theta(d, 0.0); // 初期解
    std::vector<double> y_vec = theta; // FISTA用中間変数
    std::vector<double> grad(d, 0.0);
    std::vector<double> temp(d, 0.0);
    std::vector<double> r(n, 0.0); // 残差
    double t = 1.0;

    for (int iter = 0; iter < max_iter_; iter++)
    {
        // r = X_aug * y_vec − Y
        cblas_dgemv(CblasColMajor, CblasNoTrans, n, d,
                    1.0, X_aug.data(), n,
                    y_vec.data(), 1,
                    0.0, r.data(), 1);
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            r[i] -= Y[i];
        }
        // grad = X_aug^T * r + lambda2_ * y_vec
        cblas_dgemv(CblasColMajor, CblasTrans, n, d,
                    1.0, X_aug.data(), n,
                    r.data(), 1,
                    0.0, grad.data(), 1);
#pragma omp parallel for
        for (int i = 0; i < d; i++)
        {
            grad[i] += lambda2_ * y_vec[i];
        }
// temp = y_vec − (1/L_const) * grad
#pragma omp parallel for
        for (int i = 0; i < d; i++)
        {
            temp[i] = y_vec[i] - (1.0 / L_const) * grad[i];
        }
        // new_theta = softThreshold(temp, lambda1_/L_const)
        std::vector<double> new_theta;
        softThreshold(temp, new_theta, lambda1_ / L_const, d);

        double t_new = (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0;
#pragma omp parallel for
        for (int i = 0; i < d; i++)
        {
            y_vec[i] = new_theta[i] + ((t - 1.0) / t_new) * (new_theta[i] - theta[i]);
        }
        // 収束判定: ||new_theta − theta||₂
        double diff_norm = 0.0;
        for (int i = 0; i < d; i++)
        {
            double diff = new_theta[i] - theta[i];
            diff_norm += diff * diff;
        }
        diff_norm = std::sqrt(diff_norm);
        if (diff_norm < tol_)
        {
            theta = new_theta;
            break;
        }
        theta = new_theta;
        t = t_new;
    }
    // 結果保存: θ = [weights; bias]
    weights_.assign(theta.begin(), theta.begin() + m);
    bias_ = theta[m];
    theta_ = theta;
}

// ── ADMM ソルバー ─────────────────────────────
// 問題: min_{θ,z} 0.5||X_augθ − Y||²₂ + 0.5λ2||θ||²₂ + λ1||z||₁  s.t. θ = z
void ElasticnetRegression::solveADMM(const double *X, int n, int m, const double *Y)
{
    int d = m + 1;
    std::vector<double> X_aug = augmentWithBias(X, n, m); // (n×d)
    // 事前計算: XtX = X_aug^T * X_aug と Xty = X_aug^T * Y
    std::vector<double> XtX(d * d, 0.0);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                d, d, n,
                1.0, X_aug.data(), n,
                X_aug.data(), n,
                0.0, XtX.data(), d);
    std::vector<double> Xty(d, 0.0);
    cblas_dgemv(CblasColMajor, CblasTrans, n, d,
                1.0, X_aug.data(), n,
                Y, 1,
                0.0, Xty.data(), 1);
    // 行列 A = XtX + (λ2 + ρ)I
    std::vector<double> A = XtX;
    for (int i = 0; i < d; i++)
    {
        A[i * d + i] += (lambda2_ + admm_rho_);
    }
    // Cholesky 分解
    int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', d, A.data(), d);
    if (info != 0)
        throw std::runtime_error("Cholesky factorization failed in ADMM");
    // 初期化
    std::vector<double> theta(d, 0.0);
    std::vector<double> z(d, 0.0);
    std::vector<double> u(d, 0.0);

    for (int iter = 0; iter < max_iter_; iter++)
    {
        // θ 更新: 解くべき線形系 A θ = Xty + ρ(z − u)
        std::vector<double> b = Xty;
#pragma omp parallel for
        for (int i = 0; i < d; i++)
        {
            b[i] += admm_rho_ * (z[i] - u[i]);
        }
        info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', d, 1, A.data(), d, b.data(), d);
        if (info != 0)
            throw std::runtime_error("Solving linear system failed in ADMM");
        theta = b;
        // z 更新: z = softThreshold(θ + u, λ1/ρ)
        std::vector<double> theta_plus_u(d, 0.0);
#pragma omp parallel for
        for (int i = 0; i < d; i++)
        {
            theta_plus_u[i] = theta[i] + u[i];
        }
        std::vector<double> z_new;
        softThreshold(theta_plus_u, z_new, lambda1_ / admm_rho_, d);
        z = z_new;
// u 更新: u = u + θ − z
#pragma omp parallel for
        for (int i = 0; i < d; i++)
        {
            u[i] += theta[i] - z[i];
        }
        // 収束判定: ||θ − z||₂
        double diff_norm = 0.0;
        for (int i = 0; i < d; i++)
        {
            double diff = theta[i] - z[i];
            diff_norm += diff * diff;
        }
        diff_norm = std::sqrt(diff_norm);
        if (diff_norm < tol_)
            break;
    }
    // 結果保存: 解 z = [weights; bias]
    weights_.assign(z.begin(), z.begin() + m);
    bias_ = z[m];
    theta_ = z;
}

// ── C++ API: fit ─────────────────────────────
void ElasticnetRegression::fit(const std::vector<double> &X, const std::vector<double> &Y, std::size_t n, std::size_t m)
{
    if (n <= 0 || m <= 0)
        throw std::invalid_argument("Invalid dimensions for X");
    if (X.size() != static_cast<size_t>(n * m) || Y.size() != static_cast<size_t>(n))
        throw std::invalid_argument("Input size does not match dimensions");
    if (solver_mode_ == ElasticNetSolverMode::FISTA)
        solveFISTA(X.data(), static_cast<int>(n), static_cast<int>(m), Y.data());
    else if (solver_mode_ == ElasticNetSolverMode::ADMM)
        solveADMM(X.data(), static_cast<int>(n), static_cast<int>(m), Y.data());
    else
        throw std::runtime_error("Unsupported solver mode");
}

// ── C++ API: predict ─────────────────────────────
std::vector<double> ElasticnetRegression::predict(const std::vector<double> &X, std::size_t n, std::size_t m)
{
    if (n <= 0 || m <= 0)
        throw std::invalid_argument("Invalid dimensions for X");
    if (m != weights_.size())
        throw std::invalid_argument("Feature dimension does not match the trained model");
    std::vector<double> X_aug = augmentWithBias(X.data(), static_cast<int>(n), static_cast<int>(m));
    int d = m + 1;
    std::vector<double> y_pred(n, 0.0);
    cblas_dgemv(CblasColMajor, CblasNoTrans, static_cast<int>(n), d,
                1.0, X_aug.data(), static_cast<int>(n),
                theta_.data(), 1,
                0.0, y_pred.data(), 1);
    return y_pred;
}

// // ── Python API: fit ─────────────────────────────
// void ElasticnetRegression::fit(pybind11::array_t<double> X_array, int n, int m,
//                                  pybind11::array_t<double> Y_array, int n_y, int m_y) {
//     auto X_req = X_array.request();
//     auto Y_req = Y_array.request();
//     if (X_req.size != static_cast<size_t>(n * m))
//         throw std::runtime_error("X size does not match provided dimensions");
//     if (Y_req.size != static_cast<size_t>(n_y * m_y))
//         throw std::runtime_error("Y size does not match provided dimensions");
//     if (n_y != n || m_y != 1)
//         throw std::runtime_error("Y must be of shape (n x 1)");
//     double* X_ptr = static_cast<double*>(X_req.ptr);
//     double* Y_ptr = static_cast<double*>(Y_req.ptr);
//     if (solver_mode_ == ElasticNetSolverMode::FISTA)
//         solveFISTA(X_ptr, n, m, Y_ptr);
//     else if (solver_mode_ == ElasticNetSolverMode::ADMM)
//         solveADMM(X_ptr, n, m, Y_ptr);
//     else
//         throw std::runtime_error("Unsupported solver mode");
// }

// ── Python API: predict ─────────────────────────────
// pybind11::array_t<double> ElasticnetRegression::predict(pybind11::array_t<double> X_array, int n, int m) {
//     auto X_req = X_array.request();
//     if (X_req.size != static_cast<size_t>(n * m))
//         throw std::runtime_error("X size does not match provided dimensions");
//     if (m != static_cast<int>(weights_.size()))
//         throw std::runtime_error("Feature dimension does not match the trained model");
//     std::vector<double> X_vec(static_cast<double*>(X_req.ptr),
//                               static_cast<double*>(X_req.ptr) + n * m);
//     std::vector<double> y_pred = predict(X_vec, n, m);
//     pybind11::array_t<double> result({n, 1});
//     auto res_req = result.request();
//     double* res_ptr = static_cast<double*>(res_req.ptr);
//     #pragma omp parallel for
//     for (int i = 0; i < n; i++) {
//         res_ptr[i] = y_pred[i];
//     }
//     return result;
// }

std::vector<double> ElasticnetRegression::get_weights() const
{
    return weights_;
}

double ElasticnetRegression::get_bias() const
{
    return bias_;
}
