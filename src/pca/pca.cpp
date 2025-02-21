#include <cblas.h>
#include <lapacke.h>
#include "pca/pca.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

PCA::PCA(int n_components) : n_components_(n_components), is_initialized_(false){}
PCA::~PCA() {}

// 入力: X は Col‐Major で n×m 行列（n: サンプル数, m: 特徴量数）
void PCA::fit(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  // 出力用のメンバ変数のサイズ設定
  mean_.resize(m, 0.0);
  components_.resize(m * n_components_, 0.0);

  is_initialized_ = true;

  // 1. 各特徴量（列）の平均値を計算
  //    外側のループは TBB で並列化し、内側（各列の n 個のサンプル和）は OpenMP で縮約
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, m),
                    [&](const tbb::blocked_range<std::size_t> &r)
                    {
                      for (std::size_t j = r.begin(); j != r.end(); ++j)
                      {
                        double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
                        for (std::size_t i = 0; i < n; ++i)
                        {
                          sum += X[j * n + i];
                        }
                        mean_[j] = sum / n;
                      }
                    });

  // 2. 中心化したデータの作成
  //    X のコピーを作成し、各列から平均を引く
  std::vector<double> centered_X(X);
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, m),
                    [&](const tbb::blocked_range<std::size_t> &r)
                    {
                      for (std::size_t j = r.begin(); j != r.end(); ++j)
                      {
#pragma omp parallel for
                        for (std::size_t i = 0; i < n; ++i)
                        {
                          centered_X[j * n + i] -= mean_[j];
                        }
                      }
                    });

  // 3. 共分散行列の計算
  //    cov = (1/(n-1)) * (X_centered^T * X_centered)
  //    X_centered: n×m (Col‐Major) → 共分散行列: m×m
  std::vector<double> cov_matrix(m * m, 0.0);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              m, m, n,
              1.0 / (n - 1),
              centered_X.data(), n,
              centered_X.data(), n,
              0.0,
              cov_matrix.data(), m);

  // 4. 固有値分解 (LAPACKE_dsyevd)
  //    固有値は昇順に返されるため、最大の固有値に対応する固有ベクトルは列インデックス m - n_components_ 以降にある
  std::vector<double> eigenvalues(m, 0.0);
  std::vector<double> eigenvectors = cov_matrix; // cov_matrix のコピーを用いて計算
  int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', m,
                            eigenvectors.data(), m,
                            eigenvalues.data());
  if (info != 0)
  {
    throw std::runtime_error("Eigen decomposition failed");
  }

  // 5. 上位 n_components_ 個の固有ベクトルを抽出
  //    出力 components_ は Col‐Major (m×n_components_) として保存
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n_components_),
                    [&](const tbb::blocked_range<std::size_t> &r)
                    {
                      for (std::size_t comp = r.begin(); comp != r.end(); ++comp)
                      {
                        std::size_t eig_col = m - n_components_ + comp;
                        for (std::size_t i = 0; i < m; ++i)
                        {
                          components_[i + comp * m] = eigenvectors[i + eig_col * m];
                        }
                      }
                    });
}

std::vector<double> PCA::transform(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  // 出力: n×n_components_ の行列 (Col‐Major)
  std::vector<double> result(n * n_components_, 0.0);

  // 1. 中心化
  std::vector<double> centered_X(X);
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, m),
                    [&](const tbb::blocked_range<std::size_t> &r)
                    {
                      for (std::size_t j = r.begin(); j != r.end(); ++j)
                      {
#pragma omp parallel for
                        for (std::size_t i = 0; i < n; ++i)
                        {
                          centered_X[j * n + i] -= mean_[j];
                        }
                      }
                    });

  // 2. 射影 (変換)
  //    result = X_centered * components_
  //    X_centered: n×m, components_: m×n_components_ → result: n×n_components_
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              n, n_components_, m,
              1.0,
              centered_X.data(), n,
              components_.data(), m,
              0.0,
              result.data(), n);

  return result;
}

std::vector<double> PCA::predict(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  return transform(X, n, m);
}

std::vector<double> PCA::fit_transform(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  fit(X, n, m);
  return transform(X, n, m);
}
