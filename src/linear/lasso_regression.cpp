#include "linear/lasso_regression.hpp"
#include <cstring>
#include <cblas.h>
#include <lapacke.h>
// #include <omp.h>
#ifdef __AVX512F__
#include <immintrin.h>  // AVX512 用
#elif defined(__AVX2__)
#include <immintrin.h>  // AVX2
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

// ── コンストラクタ ─────────────────────────────────────────────
LassoRegression::LassoRegression(double lambda, int max_iter, double tol,
                                 LassoSolverMode mode, double admm_rho, bool penalize_bias)
    : lambda_(lambda), max_iter_(max_iter), tol_(tol),
      solver_mode_(mode), admm_rho_(admm_rho), penalize_bias_(penalize_bias),
      bias_(0.0)
{
    if (lambda_ < 0)
        throw std::invalid_argument("lambda must be >= 0");
    if (max_iter_ <= 0)
        throw std::invalid_argument("max_iter must be > 0");
    if (tol_ <= 0)
        throw std::invalid_argument("tol must be > 0");
}

// ── 内部関数 ─────────────────────────────────────────────
// X: (n×m) → 付加後: (n×(m+1))（Col‐Major 配列）
std::vector<double> LassoRegression::augmentWithBias(const double* X, int n, int m) {
    int d = m + 1;
    std::vector<double> X_aug(n * d, 0.0);
    // もとの m 列を TBB を用いて並列コピー
    tbb::parallel_for(0, m, [=, &X_aug](int j) {
        std::memcpy(&X_aug[j * n], &X[j * n], n * sizeof(double));
    });
    // バイアス列（最後の列）を 1.0 に設定（OpenMP で並列化）
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        X_aug[m * n + i] = 1.0;
    }
    return X_aug;
}

// X_aug (n×d) の X_aug^T X_aug の最大固有値を LAPACK により計算
double LassoRegression::computeLipschitzConstant(const std::vector<double>& X_aug, int n, int d) {
    std::vector<double> A(d * d, 0.0);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                d, d, n,
                1.0,
                X_aug.data(), n,
                X_aug.data(), n,
                0.0,
                A.data(), d);
    std::vector<double> eigvals(d, 0.0);
    std::vector<double> A_copy = A;
    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'U', d, A_copy.data(), d, eigvals.data());
    if (info != 0)
        throw std::runtime_error("Eigenvalue computation failed in computeLipschitzConstant");
    return *std::max_element(eigvals.begin(), eigvals.end());
}

// softThreshold: 各要素に対して prox(x,α)= sign(x)·max(|x|−α,0) を適用（ただし、最後の要素は penalize_bias_ が false の場合はそのまま）
void LassoRegression::softThreshold(const std::vector<double>& x, std::vector<double>& out, double threshold, int d) {
    out.resize(x.size());
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        double val = x[i];
        if (!penalize_bias_ && i == d - 1) {
            out[i] = val;
        } else {
            if (val > threshold)
                out[i] = val - threshold;
            else if (val < -threshold)
                out[i] = val + threshold;
            else
                out[i] = 0.0;
        }
    }
}

// ── FISTA ソルバー ─────────────────────────────────────────────
void LassoRegression::solveFISTA(const double* X, int n, int m, const double* Y) {
    int d = m + 1;
    std::vector<double> X_aug = augmentWithBias(X, n, m);  // (n×d)
    double L_const = computeLipschitzConstant(X_aug, n, d);

    std::vector<double> theta(d, 0.0);    // 解の初期値
    std::vector<double> y_vec = theta;    // FISTA 用中間変数
    std::vector<double> grad(d, 0.0);
    std::vector<double> temp(d, 0.0);
    std::vector<double> r(n, 0.0);          // 残差
    double t = 1.0;

    for (int iter = 0; iter < max_iter_; iter++) {
        // r = X_aug * y_vec − Y
        cblas_dgemv(CblasColMajor, CblasNoTrans, n, d,
                    1.0, X_aug.data(), n,
                    y_vec.data(), 1,
                    0.0, r.data(), 1);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            r[i] -= Y[i];
        }
        // grad = X_aug^T * r
        cblas_dgemv(CblasColMajor, CblasTrans, n, d,
                    1.0, X_aug.data(), n,
                    r.data(), 1,
                    0.0, grad.data(), 1);
        // temp = y_vec − (1/L_const)*grad
        #pragma omp parallel for
        for (int i = 0; i < d; i++) {
            temp[i] = y_vec[i] - (1.0 / L_const) * grad[i];
        }
        // new_theta = softThreshold(temp, lambda_/L_const)
        std::vector<double> new_theta;
        softThreshold(temp, new_theta, lambda_ / L_const, d);

        double t_new = (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0;
        #pragma omp parallel for
        for (int i = 0; i < d; i++) {
            y_vec[i] = new_theta[i] + ((t - 1.0) / t_new) * (new_theta[i] - theta[i]);
        }
        // 収束判定: ||new_theta - theta||₂
        double diff_norm = 0.0;
        for (int i = 0; i < d; i++) {
            double diff = new_theta[i] - theta[i];
            diff_norm += diff * diff;
        }
        diff_norm = std::sqrt(diff_norm);
        if (diff_norm < tol_) {
            theta = new_theta;
            break;
        }
        theta = new_theta;
        t = t_new;
    }
    // 結果を保存: θ = [weights; bias]
    weights_.assign(theta.begin(), theta.begin() + m);
    bias_ = theta[m];
    theta_ = theta;
}

// ── ADMM ソルバー ─────────────────────────────────────────────
void LassoRegression::solveADMM(const double* X, int n, int m, const double* Y) {
    int d = m + 1;
    std::vector<double> X_aug = augmentWithBias(X, n, m);  // (n×d)
    // 事前計算: XtX = X_aug^T * X_aug
    std::vector<double> XtX(d * d, 0.0);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                d, d, n,
                1.0, X_aug.data(), n,
                X_aug.data(), n,
                0.0, XtX.data(), d);
    // 事前計算: Xty = X_aug^T * Y
    std::vector<double> Xty(d, 0.0);
    cblas_dgemv(CblasColMajor, CblasTrans, n, d,
                1.0, X_aug.data(), n,
                Y, 1,
                0.0, Xty.data(), 1);
    // 行列 A = XtX + rho I
    std::vector<double> A = XtX;
    for (int i = 0; i < d; i++) {
        A[i * d + i] += admm_rho_;
    }
    // Cholesky 分解
    int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', d, A.data(), d);
    if (info != 0)
        throw std::runtime_error("Cholesky factorization failed in ADMM");
    // 初期化
    std::vector<double> theta(d, 0.0);
    std::vector<double> z(d, 0.0);
    std::vector<double> u(d, 0.0);

    for (int iter = 0; iter < max_iter_; iter++) {
        // theta 更新: 解くべき線形系 A*theta = Xty + rho*(z - u)
        std::vector<double> b = Xty;
        #pragma omp parallel for
        for (int i = 0; i < d; i++) {
            b[i] += admm_rho_ * (z[i] - u[i]);
        }
        info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', d, 1, A.data(), d, b.data(), d);
        if (info != 0)
            throw std::runtime_error("Solving linear system failed in ADMM");
        theta = b;
        // z 更新: z = softThreshold(theta + u, lambda/rho)
        std::vector<double> theta_plus_u(d, 0.0);
        #pragma omp parallel for
        for (int i = 0; i < d; i++) {
            theta_plus_u[i] = theta[i] + u[i];
        }
        std::vector<double> z_new;
        softThreshold(theta_plus_u, z_new, lambda_ / admm_rho_, d);
        z = z_new;
        // u 更新: u = u + theta - z
        #pragma omp parallel for
        for (int i = 0; i < d; i++) {
            u[i] += theta[i] - z[i];
        }
        // 収束判定: ||theta - z||₂
        double diff_norm = 0.0;
        for (int i = 0; i < d; i++) {
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

// ── C++ API: fit ─────────────────────────────────────────────
void LassoRegression::fit(const std::vector<double>& X, const std::vector<double>& Y, std::size_t n, std::size_t m) {
    if (n <= 0 || m <= 0)
        throw std::invalid_argument("Invalid dimensions for X");
    if (X.size() != static_cast<size_t>(n * m) || Y.size() != static_cast<size_t>(n))
        throw std::invalid_argument("Input size does not match dimensions");
    if (solver_mode_ == LassoSolverMode::FISTA)
        solveFISTA(X.data(), static_cast<int>(n), static_cast<int>(m), Y.data());
    else if (solver_mode_ == LassoSolverMode::ADMM)
        solveADMM(X.data(), static_cast<int>(n), static_cast<int>(m), Y.data());
    else
        throw std::runtime_error("Unsupported solver mode");
}

// ── C++ API: predict ─────────────────────────────────────────
std::vector<double> LassoRegression::predict(const std::vector<double>& X, std::size_t n, std::size_t m) {
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

// ── Python API: fit ──────────────────────────────────────────
// void LassoRegression::fit(pybind11::array_t<double> X_array, int n, int m,
//                           pybind11::array_t<double> Y_array, int n_y, int m_y) {
//     auto X_req = X_array.request();
//     auto Y_req = Y_array.request();
//     if (X_req.size != static_cast<size_t>(n * m))
//         throw std::runtime_error("X size does not match provided dimensions");
//     if (Y_req.size != static_cast<size_t>(n_y * m_y))
//         throw std::runtime_error("Y size does not match provided dimensions");
//     if (n_y != n || m_y != 1)
//         throw std::runtime_error("Y must be of shape (n×1)");
//     double* X_ptr = static_cast<double*>(X_req.ptr);
//     double* Y_ptr = static_cast<double*>(Y_req.ptr);
//     if (solver_mode_ == LassoSolverMode::FISTA)
//         solveFISTA(X_ptr, n, m, Y_ptr);
//     else if (solver_mode_ == LassoSolverMode::ADMM)
//         solveADMM(X_ptr, n, m, Y_ptr);
//     else
//         throw std::runtime_error("Unsupported solver mode");
// }

// ── Python API: predict ──────────────────────────────────────
// pybind11::array_t<double> LassoRegression::predict(pybind11::array_t<double> X_array, int n, int m) {
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

std::vector<double> LassoRegression::get_weights() const {
    return weights_;
}

double LassoRegression::get_bias() const {
    return bias_;
}
