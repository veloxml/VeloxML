#include "linear/ridge_regression.hpp"
#include <cstring>   // memcpy
#include <cblas.h>   // BLAS
#include <lapacke.h> // LAPACK
#include <omp.h>     // OpenMP
#ifdef __AVX512F__
#include <immintrin.h> // AVX512 用
#elif defined(__AVX2__)
#include <immintrin.h> // AVX2
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// --- コンストラクタ ---
RidgeRegression::RidgeRegression(double lambda, bool penalize_bias)
    : lambda_(lambda), penalize_bias_(penalize_bias), bias_(0.0)
{
    if (lambda_ <= 0.0)
        throw std::invalid_argument("lambda must be > 0");
}

// --- 入力行列にバイアス項を付加する ---
// X: (n x m) Col-Major 配列 → X_aug: (n x (m+1)) (各列が連続)
std::vector<double> RidgeRegression::augmentWithBias(const double *X, int n, int m)
{
    int m_aug = m + 1;
    std::vector<double> X_aug(n * m_aug, 0.0);
    // もとの m 列を TBB で並列コピー
    tbb::parallel_for(0, m, [=, &X_aug](int j)
                      { std::memcpy(&X_aug[j * n], &X[j * n], n * sizeof(double)); });
// 最後の列（バイアス）を1.0で埋める（OpenMPで並列化）
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        X_aug[m * n + i] = 1.0;
    }
    return X_aug;
}

// --- 対角要素に正則化項を加える ---
// SIMD最適化として、AVX512/AVX2/NEONに対応（対象はAの対角成分：アドレスは [i*(dim+1)]）
void RidgeRegression::applyRegularization(double *A, int dim, double lambda, bool penalize_bias)
{
    // 最後の対角要素はバイアス項として、penalize_bias が false なら変更しない
    int limit = penalize_bias ? dim : (dim - 1);
#ifdef __AVX512F__
    int i = 0;
    for (; i <= limit - 8; i += 8)
    {
        __m512d reg = _mm512_set1_pd(lambda);
        // インデックスを計算（各対角成分のオフセットは i*(dim+1)）
        alignas(64) int idx[8];
        for (int j = 0; j < 8; j++)
        {
            idx[j] = (i + j) * (dim + 1);
        }
        __m512i index = _mm512_load_si512(idx);
        __m512d diag = _mm512_i32gather_pd(index, A, 8);
        diag = _mm512_add_pd(diag, reg);
        _mm512_i32scatter_pd(A, index, diag, 8);
    }
    for (; i < limit; i++)
    {
        A[i * (dim + 1)] += lambda;
    }
#elif defined(__AVX2__)
    int i = 0;
    for (; i <= limit - 4; i += 4)
    {
        __m256d reg = _mm256_set1_pd(lambda);
        double diag[4];
        for (int j = 0; j < 4; j++)
        {
            diag[j] = A[(i + j) * (dim + 1)];
        }
        __m256d d = _mm256_loadu_pd(diag);
        d = _mm256_add_pd(d, reg);
        _mm256_storeu_pd(diag, d);
        for (int j = 0; j < 4; j++)
        {
            A[(i + j) * (dim + 1)] = diag[j];
        }
    }
    for (; i < limit; i++)
    {
        A[i * (dim + 1)] += lambda;
    }
#elif defined(__ARM_NEON)
// ARM NEON の double 精度サポートは限定的なため、スカラーで処理
#pragma omp parallel for simd
    for (int i = 0; i < limit; i++)
    {
        A[i * (dim + 1)] += lambda;
    }
#else
#pragma omp parallel for simd
    for (int i = 0; i < limit; i++)
    {
        A[i * (dim + 1)] += lambda;
    }
#endif
}

// --- Ridge回帰のパラメータ計算 ---
// 解く対象は、(X_aug^T * X_aug + λI) θ = X_aug^T Y
// ※X_augはバイアス列付き (n x (m+1)) 行列
void RidgeRegression::computeRegression(const double *X, int n, int m, const double *Y)
{
    int m_aug = m + 1;
    // X_aug を作成
    std::vector<double> X_aug = augmentWithBias(X, n, m);

    // A = X_aug^T * X_aug を計算（サイズ: m_aug x m_aug）
    std::vector<double> A(m_aug * m_aug, 0.0);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m_aug, m_aug, n,
                1.0,
                X_aug.data(), n,
                X_aug.data(), n,
                0.0,
                A.data(), m_aug);

    // B = X_aug^T * Y（サイズ: m_aug）
    std::vector<double> B(m_aug, 0.0);
    cblas_dgemv(CblasColMajor, CblasTrans, n, m_aug,
                1.0,
                X_aug.data(), n,
                Y, 1,
                0.0,
                B.data(), 1);

    // 正則化項を加える（バイアス項は penalize_bias_ が false なら除外）
    applyRegularization(A.data(), m_aug, lambda_, penalize_bias_);

    // A は対称正定値となるため、Cholesky分解により解く
    int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', m_aug, A.data(), m_aug);
    if (info != 0)
        throw std::runtime_error("Cholesky factorization failed in ridge regression");
    info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', m_aug, 1, A.data(), m_aug, B.data(), m_aug);
    if (info != 0)
        throw std::runtime_error("Solving linear system failed in ridge regression");

    // 得られた解 B には θ が格納され、先頭 m 成分が重み、最後がバイアス
    weights_.assign(B.begin(), B.begin() + m);
    bias_ = B[m];
}

// --- C++ API: fit ---
// X: (n x m) のCol-Major配列、Y: (n x 1)
void RidgeRegression::fit(const std::vector<double> &X, const std::vector<double> &Y, std::size_t n, std::size_t m)
{
    if (n <= 0 || m <= 0)
        throw std::invalid_argument("Invalid dimensions for X");

        if (X.size() != static_cast<size_t>(n * m) || Y.size() != static_cast<size_t>(n))
        throw std::invalid_argument("Input size does not match dimensions");
    computeRegression(X.data(), static_cast<int>(n), static_cast<int>(m), Y.data());
}

// --- C++ API: predict ---
// 入力 X: (n x m) のCol-Major配列 → 出力: (n x 1) の予測値ベクトル
std::vector<double> RidgeRegression::predict(const std::vector<double> &X, std::size_t n, std::size_t m)
{
    if (n <= 0 || m <= 0)
        throw std::invalid_argument("Invalid dimensions for X");
    if (m != weights_.size())
        throw std::invalid_argument("Feature dimension does not match the trained model");
    // X_aug を作成
    std::vector<double> X_aug = augmentWithBias(X.data(), static_cast<int>(n), static_cast<int>(m));
    int m_aug = m + 1;
    // θ = [weights, bias]
    std::vector<double> theta(m_aug);
    std::copy(weights_.begin(), weights_.end(), theta.begin());
    theta[m] = bias_;
    // y_pred = X_aug * θ
    std::vector<double> y_pred(n, 0.0);
    cblas_dgemv(CblasColMajor, CblasNoTrans, static_cast<int>(n), m_aug,
                1.0,
                X_aug.data(), static_cast<int>(n),
                theta.data(), 1,
                0.0,
                y_pred.data(), 1);
    return y_pred;
}

std::vector<double> RidgeRegression::get_weights() const
{
    return weights_;
}

double RidgeRegression::get_bias() const
{
    return bias_;
}
