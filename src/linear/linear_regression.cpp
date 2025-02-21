#include "linear/linear_regression.hpp"
#include <cstring>   // memcpy
#include <cblas.h>   // BLAS ルーチン
#include <lapacke.h> // LAPACK ルーチン
#include <omp.h>     // OpenMP
#include <stdexcept>
#include <string>
#include <vector>
#include <tbb/parallel_for.h>

// --- コンストラクタ ---
LinearRegression::LinearRegression(LinearDecompositionMode mode)
    : mode_(mode), bias_(0.0)
{
}

LinearRegression::~LinearRegression() {};

// --- 内部関数 ---
// X (n x m) の行列にバイアス列を付加して (n x (m+1)) の配列を返す
// 改善点: 毎回新たにメモリを確保するのではなく、メンババッファ X_aug_buffer_ を再利用する。
const std::vector<double> &LinearRegression::augmentWithBias(const double *X, int n, int m)
{
    int m_aug = m + 1;
    size_t size = static_cast<size_t>(n) * m_aug;
    if (X_aug_buffer_.size() != size)
    {
        X_aug_buffer_.resize(size);
    }
    // 並列化の粒度調整: 特徴量数が十分大きい場合のみ TBB を使用
    if (m >= 10)
    {
        tbb::parallel_for(0, m, [=](int j)
                          { std::memcpy(&X_aug_buffer_[j * n], &X[j * n], n * sizeof(double)); });
    }
    else
    {
        for (int j = 0; j < m; ++j)
        {
            std::memcpy(&X_aug_buffer_[j * n], &X[j * n], n * sizeof(double));
        }
    }
    // バイアス列を設定（OpenMP simd を利用）
#pragma omp simd
    for (int i = 0; i < n; i++)
    {
        X_aug_buffer_[m * n + i] = 1.0;
    }
    return X_aug_buffer_;
}

// 回帰パラメータ（theta）を計算する内部関数
// 正規方程式： (X_aug^T * X_aug) * theta = (X_aug^T * Y) を各分解モードに応じて解く
void LinearRegression::computeRegression(const double *X, int n, int m, const double *Y)
{
    int m_aug = m + 1;
    // 1. バイアス項を付加した入力行列 X_aug を作成 (サイズ: n x (m+1))
    const std::vector<double> &X_aug = augmentWithBias(X, n, m);

    if (mode_ == LinearDecompositionMode::LU)
    {
        // --- LU (Cholesky) モード ---
        std::vector<double> A(m_aug * m_aug, 0.0);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m_aug, m_aug, n,
                    1.0,
                    X_aug.data(), n,
                    X_aug.data(), n,
                    0.0,
                    A.data(), m_aug);

        std::vector<double> B(m_aug, 0.0);
        cblas_dgemv(CblasColMajor, CblasTrans, n, m_aug,
                    1.0,
                    X_aug.data(), n,
                    Y, 1,
                    0.0,
                    B.data(), 1);

        int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', m_aug, A.data(), m_aug);
        if (info != 0)
        {
            throw std::runtime_error("Cholesky 分解に失敗しました (LAPACKE_dpotrf returned " + std::to_string(info) + ")");
        }
        info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', m_aug, 1, A.data(), m_aug, B.data(), m_aug);
        if (info != 0)
        {
            throw std::runtime_error("線形方程式の解法に失敗しました (LAPACKE_dpotrs returned " + std::to_string(info) + ")");
        }
        weights_.assign(B.begin(), B.begin() + m);
        bias_ = B[m];
    }
    else if (mode_ == LinearDecompositionMode::QR)
    {
        // --- QR モード ---
        int d = m_aug;
        std::vector<double> Q = X_aug; // n x d
        std::vector<double> tau(d, 0.0);
        int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, d, Q.data(), n, tau.data());
        if (info != 0)
        {
            throw std::runtime_error("QR 分解に失敗しました (LAPACKE_dgeqrf returned " + std::to_string(info) + ")");
        }
        // Q^T * Y を計算
        std::vector<double> b(n, 0.0);
        std::copy(Y, Y + n, b.begin());
        info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, 1, d, Q.data(), n, tau.data(), b.data(), n);
        if (info != 0)
        {
            throw std::runtime_error("Q^T * Y の計算に失敗しました (LAPACKE_dormqr returned " + std::to_string(info) + ")");
        }
        // R は Q の上三角部 (n x d の上 d x d 部分)
        std::vector<double> R(d * d, 0.0);
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i <= j; i++)
            {
                R[i + j * d] = Q[j * n + i];
            }
        }
        info = LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', d, 1, R.data(), d, b.data(), d);
        if (info != 0)
        {
            throw std::runtime_error("上三角系の解法に失敗しました (LAPACKE_dtrtrs returned " + std::to_string(info) + ")");
        }
        std::vector<double> theta = b; // d 要素
        weights_.assign(theta.begin(), theta.begin() + m);
        bias_ = theta[m];
    }
    else if (mode_ == LinearDecompositionMode::SVD)
    {
        // --- SVD モード ---
        int d = m_aug;
        std::vector<double> U(n * d, 0.0);
        std::vector<double> S(d, 0.0);
        std::vector<double> VT(d * d, 0.0);
        std::vector<double> X_copy = X_aug; // SVD は上書きするためコピー
        std::vector<double> superb(d - 1, 0.0);
        int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', n, d,
                                  X_copy.data(), n, S.data(),
                                  U.data(), n, VT.data(), d,
                                  superb.data());
        if (info != 0)
        {
            throw std::runtime_error("SVD に失敗しました (LAPACKE_dgesvd returned " + std::to_string(info) + ")");
        }
        // U^T * Y の計算 (サイズ: d)
        std::vector<double> UTY(d, 0.0);
        cblas_dgemv(CblasColMajor, CblasTrans, n, d,
                    1.0,
                    U.data(), n,
                    Y, 1,
                    0.0,
                    UTY.data(), 1);
        // 特異値に対して逆数を掛ける（閾値 1e-12 以下は 0）
        for (int i = 0; i < d; i++)
        {
            double invS = (S[i] > 1e-12 ? 1.0 / S[i] : 0.0);
            UTY[i] *= invS;
        }
        // theta = V * (U^T * Y) ＝ V * UTY
        std::vector<double> theta(d, 0.0);
        // 手動ループ (dgemv でも可)
        for (int i = 0; i < d; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < d; j++)
            {
                // VT は V^T なので、V(i,j) = VT[j + i*d]
                sum += VT[j + i * d] * UTY[j];
            }
            theta[i] = sum;
        }
        weights_.assign(theta.begin(), theta.begin() + m);
        bias_ = theta[m];
    }
    else
    {
        throw std::runtime_error("未対応の分解モード");
    }
}

// --- C++ API ---
// fit: 1D の Col-Major 配列 X, Y として入力。Y は (n x 1) のベクトルとする。
void LinearRegression::fit(const std::vector<double>& X, const std::vector<double>& Y, std::size_t n, std::size_t m)
{
    if (n <= 0 || m <= 0)
    {
        throw std::invalid_argument("X の次元が不正です");
    }
    if (X.size() != static_cast<size_t>(n * m) || Y.size() != static_cast<size_t>(n))
    {
        throw std::invalid_argument("入力サイズが次元と一致しません");
    }
    computeRegression(X.data(), static_cast<int>(n), static_cast<int>(m), Y.data());
}

// predict: 1D の Col-Major 配列 X を入力し、(n x 1) の予測値ベクトルを返す
std::vector<double> LinearRegression::predict(const std::vector<double> &X, std::size_t n, std::size_t m)
{
    if (n <= 0 || m <= 0)
    {
        throw std::invalid_argument("X の次元が不正です");
    }
    if (m != weights_.size())
    {
        throw std::invalid_argument("特徴量の次元が学習済みモデルと一致しません");
    }
    const std::vector<double> &X_aug = augmentWithBias(X.data(), static_cast<int>(n), static_cast<int>(m));
    int m_aug = m + 1;

    // theta = [weights, bias] の作成
    std::vector<double> theta(m_aug);
    std::copy(weights_.begin(), weights_.end(), theta.begin());
    theta[m] = bias_;

    // y_pred = X_aug * theta を BLAS の dgemv で計算
    std::vector<double> y_pred(n, 0.0);
    cblas_dgemv(CblasColMajor, CblasNoTrans, static_cast<int>(n), m_aug,
                1.0,
                X_aug.data(), static_cast<int>(n),
                theta.data(), 1,
                0.0,
                y_pred.data(), 1);
    return y_pred;
}

// --- Python API ---
// 入力は py::array_t<double>（1D Col-Major 配列）とし、行数・列数を引数で指定する。
// void LinearRegression::fit(pybind11::array_t<double> X_array, int n, int m,
//                            pybind11::array_t<double> Y_array, int n_y, int m_y)
// {
//     auto X_req = X_array.request();
//     auto Y_req = Y_array.request();
//     if (X_req.size != static_cast<size_t>(n * m))
//         throw std::runtime_error("X のサイズが指定された次元と一致しません");
//     if (Y_req.size != static_cast<size_t>(n_y * m_y))
//         throw std::runtime_error("Y のサイズが指定された次元と一致しません");
//     if (n_y != n || m_y != 1)
//         throw std::runtime_error("Y は (n x 1) の列ベクトルでなければなりません");

//     double *X_ptr = static_cast<double *>(X_req.ptr);
//     double *Y_ptr = static_cast<double *>(Y_req.ptr);
//     computeRegression(X_ptr, n, m, Y_ptr);
// }

// pybind11::array_t<double> LinearRegression::predict(pybind11::array_t<double> X_array, int n, int m)
// {
//     auto X_req = X_array.request();
//     if (X_req.size != static_cast<size_t>(n * m))
//         throw std::runtime_error("X のサイズが指定された次元と一致しません");
//     if (m != static_cast<int>(weights_.size()))
//         throw std::runtime_error("特徴量の次元が学習済みモデルと一致しません");

//     double *X_ptr = static_cast<double *>(X_req.ptr);
//     const std::vector<double> &X_aug = augmentWithBias(X_ptr, n, m);
//     int m_aug = m + 1;

//     std::vector<double> theta(m_aug);
//     std::copy(weights_.begin(), weights_.end(), theta.begin());
//     theta[m] = bias_;

//     std::vector<double> y_pred(n, 0.0);
//     cblas_dgemv(CblasColMajor, CblasNoTrans, n, m_aug,
//                 1.0,
//                 X_aug.data(), n,
//                 theta.data(), 1,
//                 0.0,
//                 y_pred.data(), 1);

//     pybind11::array_t<double> result({n, 1});
//     auto res_req = result.request();
//     double *res_ptr = static_cast<double *>(res_req.ptr);
// #pragma omp parallel for
//     for (int i = 0; i < n; i++)
//     {
//         res_ptr[i] = y_pred[i];
//     }
//     return result;
// }

std::vector<double> LinearRegression::get_weights() const
{
    return weights_;
}

double LinearRegression::get_bias() const
{
    return bias_;
}
