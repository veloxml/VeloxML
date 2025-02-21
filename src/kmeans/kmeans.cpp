// File: KMeans.cpp
#include "kmeans/kmeans.hpp"
#include "kdtree/kdtree.hpp"
#include <cmath>
#include <cstdlib>
#include <random>
#include <limits>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cblas.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/spin_mutex.h>
#include <atomic>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// ----------------------------------------------------------------------
// 補助関数: 2点間のユークリッド距離の二乗（m は次元数）
// a, b は連続した m 要素の配列とする
// ----------------------------------------------------------------------
double KMeans::sq_euclidean_distance(const double *a, const double *b, std::size_t m)
{
  double sum = 0.0;

#if defined(__AVX512F__)
  __m512d sum_v = _mm512_setzero_pd();
  std::size_t i = 0;
  for (; i + 8 <= m; i += 8)
  {
    __m512d va = _mm512_loadu_pd(a + i);
    __m512d vb = _mm512_loadu_pd(b + i);
    __m512d diff = _mm512_sub_pd(va, vb);
    sum_v = _mm512_fmadd_pd(diff, diff, sum_v);
  }
  sum += _mm512_reduce_add_pd(sum_v);
#elif defined(__AVX2__)
  __m256d sum_v = _mm256_setzero_pd();
  std::size_t i = 0;
  for (; i + 4 <= m; i += 4)
  {
    __m256d va = _mm256_loadu_pd(a + i);
    __m256d vb = _mm256_loadu_pd(b + i);
    __m256d diff = _mm256_sub_pd(va, vb);
    sum_v = _mm256_fmadd_pd(diff, diff, sum_v);
  }
  double buf[4];
  _mm256_storeu_pd(buf, sum_v);
  sum += buf[0] + buf[1] + buf[2] + buf[3];
#elif defined(__ARM_NEON)
  float64x2_t sum_v = vdupq_n_f64(0.0);
  std::size_t i = 0;
  for (; i + 2 <= m; i += 2)
  {
    float64x2_t va = vld1q_f64(a + i);
    float64x2_t vb = vld1q_f64(b + i);
    float64x2_t diff = vsubq_f64(va, vb);
    sum_v = vfmaq_f64(sum_v, diff, diff);
  }
  double buf[2];
  vst1q_f64(buf, sum_v);
  sum += buf[0] + buf[1];
#endif

  // 残りの要素を通常のループで計算
  for (std::size_t i = m - (m % 8); i < m; ++i)
  {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

// ----------------------------------------------------------------------
// コンストラクタ／デストラクタ
// ----------------------------------------------------------------------
KMeans::KMeans(int n_clusters, int max_iter, double tol,
               KMeansAlgorithm algorithm, bool use_kdtree)
    : n_clusters_(n_clusters), max_iter_(max_iter), tol_(tol),
      algorithm_(algorithm), use_kdtree_(use_kdtree), is_initialized_(false)
{
  // centroids_ は (n_clusters_ x m) の Col-Major 配列となる（初期化は fit 時）
}

KMeans::~KMeans()
{
}

// ----------------------------------------------------------------------
// クラスタ中心の初期化 (k-means++ を用いる)
// X: 入力データ (n × m, Col-Major)、n: サンプル数, m: 特徴量数
// ----------------------------------------------------------------------
void KMeans::initialize_centroids(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  // centroids_ を (n_clusters_ * m) のサイズにリサイズ（Col‑Major形式）
  centroids_.resize(n_clusters_ * m);

  // 乱数生成器の初期化
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<std::size_t> dis(0, n - 1);

  // 1. 最初のクラスタ中心をランダムなサンプルから選択
  std::size_t first_index = dis(gen);
  for (std::size_t j = 0; j < m; j++)
  {
    // サンプル first_index の j 番目の特徴量: X[j * n + first_index]
    centroids_[j * n_clusters_ + 0] = X[j * n + first_index];
  }

  // 各サンプルの、既に選ばれた中心までの最小二乗距離を保持する配列
  std::vector<double> distances(n, std::numeric_limits<double>::max());

  // 1番目の中心から各サンプルまでの距離を並列計算
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                    [&](const tbb::blocked_range<size_t> &r)
                    {
                      for (size_t i = r.begin(); i < r.end(); i++)
                      {
                        double d = 0.0;
#pragma omp simd reduction(+ : d) simdlen(8)
                        for (std::size_t j = 0; j < m; j++)
                        {
                          double diff = X[j * n + i] - centroids_[j * n_clusters_ + 0];
                          d += diff * diff;
                        }
                        distances[i] = d;
                      }
                    });

  // 2. 残りのクラスタ中心を順次選択
  for (int k = 1; k < n_clusters_; k++)
  {
    // 距離の総和を TBB の parallel_reduce で計算
    double total_distance = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n),
        0.0,
        [&](const tbb::blocked_range<size_t> &r, double init) -> double
        {
          for (size_t i = r.begin(); i < r.end(); i++)
          {
            init += distances[i];
          }
          return init;
        },
        std::plus<double>());

    // 重み付きランダムサンプリング: 合計距離に比例して新しい中心を選ぶ
    std::uniform_real_distribution<double> dis_real(0, total_distance);
    double rand_val = dis_real(gen);
    double cumulative = 0.0;
    std::size_t next_index = 0;
    for (std::size_t i = 0; i < n; i++)
    {
      cumulative += distances[i];
      if (cumulative >= rand_val)
      {
        next_index = i;
        break;
      }
    }

    // 次の中心として、サンプル next_index を設定
    for (std::size_t j = 0; j < m; j++)
    {
      centroids_[j * n_clusters_ + k] = X[j * n + next_index];
    }

    // 各サンプルについて、新たに選ばれた中心との距離を計算し、
    // 既存の distances と比較して小さい方を保持する（並列化）
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const tbb::blocked_range<size_t> &r)
                      {
                        for (size_t i = r.begin(); i < r.end(); i++)
                        {
                          double d_new = 0.0;
#pragma omp simd reduction(+ : d_new) simdlen(8)
                          for (std::size_t j = 0; j < m; j++)
                          {
                            double diff = X[j * n + i] - X[j * n + next_index];
                            d_new += diff * diff;
                          }
                          if (d_new < distances[i])
                          {
                            distances[i] = d_new;
                          }
                        }
                      });
  }
}

// ----------------------------------------------------------------------
// run_standard: 標準の Lloyd's algorithm
// ----------------------------------------------------------------------
void KMeans::run_standard(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  // サンプルごとのクラスタ割り当て（0-indexed）
  std::vector<int> labels(n, -1);

  for (int iter = 0; iter < max_iter_; iter++)
  {
    bool changed = false;

// ===== 割り当てステップ =====
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; i++)
    {
      int best_cluster = -1;
      double best_dist = std::numeric_limits<double>::max();
      // 各クラスタとの距離を計算
      for (int k = 0; k < n_clusters_; k++)
      {
        double d = 0.0;
        for (std::size_t j = 0; j < m; j++)
        {
          double diff = X[j * n + i] - centroids_[j * n_clusters_ + k];
          d += diff * diff;
        }
        if (d < best_dist)
        {
          best_dist = d;
          best_cluster = k;
        }
      }
      // 更新があれば記録
      if (labels[i] != best_cluster)
      {
        labels[i] = best_cluster;
        changed = true;
      }
    }

    // 収束していれば終了
    if (!changed)
      break;

    // ===== 更新ステップ =====
    std::vector<double> new_centroids(n_clusters_ * m, 0.0);
    std::vector<int> counts(n_clusters_, 0);

    // 各サンプルについて、対応するクラスタ中心の和を計算
    for (std::size_t i = 0; i < n; i++)
    {
      int k = labels[i];
      counts[k]++;
      for (std::size_t j = 0; j < m; j++)
      {
        new_centroids[j * n_clusters_ + k] += X[j * n + i];
      }
    }

    // 平均をとる
    for (int k = 0; k < n_clusters_; k++)
    {
      if (counts[k] > 0)
      {
        double inv = 1.0 / counts[k];
        for (std::size_t j = 0; j < m; j++)
        {
          new_centroids[j * n_clusters_ + k] *= inv;
        }
      }
      else
      {
        // サンプルが割り当てられていないクラスタは、ランダムなサンプルで再初期化
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::size_t> dis(0, n - 1);
        std::size_t rand_index = dis(gen);
        for (std::size_t j = 0; j < m; j++)
        {
          new_centroids[j * n_clusters_ + k] = X[j * n + rand_index];
        }
      }
    }

    // 収束判定：各クラスタ中心の移動距離の合計が tol_ 未満なら終了
    double shift = 0.0;
    for (int k = 0; k < n_clusters_; k++)
    {
      double d = 0.0;
      for (std::size_t j = 0; j < m; j++)
      {
        double diff = centroids_[j * n_clusters_ + k] - new_centroids[j * n_clusters_ + k];
        d += diff * diff;
      }
      shift += std::sqrt(d);
    }
    centroids_ = new_centroids;
    if (shift < tol_)
      break;
  }
}

// ----------------------------------------------------------------------
// run_elkan: Elkan’s Algorithm
// ----------------------------------------------------------------------
// run_elkan: Elkan’s Algorithm の実装（Col-Major 入力、内部centroids_もCol-Major）
// 最適化: SIMD バッチ処理, ループアンローリング, BLAS (dgemm, dscal) の利用, TBB 並列化、境界更新精度の向上
void KMeans::run_elkan(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  // ----------------------------
  // 初期割り当て: 各サンプルについて全クラスタとの距離を計算し、上界 u と下界 l を初期化
  // ----------------------------
  std::vector<int> labels(n, -1);
  std::vector<double> u(n, std::numeric_limits<double>::max());
  // 下界 l: 各サンプルについて各クラスタの距離 (サイズ: n x n_clusters_)
  std::vector<double> l(n * n_clusters_, 0.0);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                    [&](const tbb::blocked_range<size_t> &r)
                    {
                      for (size_t i = r.begin(); i < r.end(); i++)
                      {
                        double best = std::numeric_limits<double>::max();
                        double second_best = std::numeric_limits<double>::max();
                        int best_idx = -1;
                        for (int k = 0; k < n_clusters_; k++)
                        {
                          double sum = 0.0;
        // 内側ループ: SIMD化のヒントを与える
#pragma omp simd reduction(+ : sum) simdlen(4)
                          for (std::size_t j = 0; j < m; j++)
                          {
                            double diff = X[j * n + i] - centroids_[j * n_clusters_ + k];
                            sum += diff * diff;
                          }
                          double d = std::sqrt(sum);
                          l[i * n_clusters_ + k] = d; // 初期下界: 実際の距離
                          if (d < best)
                          {
                            second_best = best;
                            best = d;
                            best_idx = k;
                          }
                          else if (d < second_best)
                          {
                            second_best = d;
                          }
                        }
                        labels[i] = best_idx;
                        u[i] = best;
                      }
                    });

  // 主反復ループ
  for (int iter = 0; iter < max_iter_; iter++)
  {

    // ----------------------------
    // 更新ステップ: 各クラスタ中心の再計算
    // ----------------------------
    // ここでは、各サンプルの所属を示すインジケータ行列 I (n x n_clusters_) を構築し、
    // その後 cblas_dgemm を用いて new_centroids = X * I を一括計算する。
    std::vector<double> I(n * n_clusters_, 0.0);
    for (size_t i = 0; i < n; i++)
    {
      int cl = labels[i];
      I[i * n_clusters_ + cl] = 1.0;
    }
    // 変換: I を row-major から col-major へ
    std::vector<double> I_cm(n * n_clusters_, 0.0);
    for (size_t i = 0; i < n; i++)
    {
      for (int k = 0; k < n_clusters_; k++)
      {
        I_cm[k * n + i] = I[i * n_clusters_ + k];
      }
    }
    // X: m x n (ColMajor), I_cm: n x n_clusters_ (ColMajor)
    // 結果 new_centroids: m x n_clusters_ (ColMajor)
    std::vector<double> new_centroids(m * n_clusters_, 0.0);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(m), n_clusters_, static_cast<int>(n),
                1.0, X.data(), static_cast<int>(m),
                I_cm.data(), static_cast<int>(n),
                0.0, new_centroids.data(), static_cast<int>(m));
    // 各クラスタの総和を得たので、各列を各クラスタのサンプル数で割る
    std::vector<int> counts(n_clusters_, 0);
    for (size_t i = 0; i < n; i++)
    {
      counts[labels[i]]++;
    }
    for (int k = 0; k < n_clusters_; k++)
    {
      if (counts[k] > 0)
      {
        double inv = 1.0 / counts[k];
        cblas_dscal(static_cast<int>(m), inv, &new_centroids[k * m], 1);
      }
      else
      {
        // ランダム再初期化
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, n - 1);
        size_t idx = dis(gen);
        for (std::size_t j = 0; j < m; j++)
        {
          new_centroids[k * m + j] = X[j * n + idx];
        }
      }
    }
    // new_centroids は row-major: 各クラスタ k is stored contiguously in block [k*m, (k+1)*m)
    // 変換して内部保持用の ColMajor に変換: element (j, k) -> index: j * n_clusters_ + k
    std::vector<double> new_centroids_cm(n_clusters_ * m, 0.0);
    for (int k = 0; k < n_clusters_; k++)
    {
      for (std::size_t j = 0; j < m; j++)
      {
        new_centroids_cm[j * n_clusters_ + k] = new_centroids[k * m + j];
      }
    }
    // 各クラスタ中心のシフト量を計算: shift[k] = norm(centroids_[*,k] - new_centroids_cm[*,k])
    std::vector<double> shift(n_clusters_, 0.0);
    for (int k = 0; k < n_clusters_; k++)
    {
      double sum = 0.0;
      for (std::size_t j = 0; j < m; j++)
      {
        double diff = centroids_[j * n_clusters_ + k] - new_centroids_cm[j * n_clusters_ + k];
        sum += diff * diff;
      }
      shift[k] = std::sqrt(sum);
    }
    centroids_ = new_centroids_cm;

    // ----------------------------
    // 安全域の計算: r[k] = 0.5 * min_{l != k} distance(centroid_k, centroid_l)
    // ----------------------------
    std::vector<double> r(n_clusters_, std::numeric_limits<double>::max());
    for (int k = 0; k < n_clusters_; k++)
    {
      for (int kk = 0; kk < n_clusters_; kk++)
      {
        if (k == kk)
          continue;
        double sum = 0.0;
#pragma omp simd reduction(+ : sum) simdlen(4)
        for (std::size_t j = 0; j < m; j++)
        {
          double diff = centroids_[j * n_clusters_ + k] - centroids_[j * n_clusters_ + kk];
          sum += diff * diff;
        }
        double d = std::sqrt(sum);
        if (d < r[k])
          r[k] = d;
      }
      r[k] *= 0.5;
    }

    // ----------------------------
    // 境界更新:
    // u[i] = u[i] + shift[labels[i]]
    // l(i,k) = max(l(i,k) - shift[k], 0)
    // ----------------------------
    for (size_t i = 0; i < n; i++)
    {
      int cl = labels[i];
      u[i] += shift[cl];
      for (int k = 0; k < n_clusters_; k++)
      {
        double new_l = l[i * n_clusters_ + k] - shift[k];
        l[i * n_clusters_ + k] = (new_l > 0 ? new_l : 0.0);
      }
    }

    // ----------------------------
    // 再割り当てステップ: 境界条件に基づいて必要な場合に各サンプルの所属クラスタを変更
    // ----------------------------
    std::atomic<bool> any_changed(false);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const tbb::blocked_range<size_t> &range)
                      {
                        for (size_t i = range.begin(); i < range.end(); i++)
                        {
                          int cl = labels[i];
                          if (u[i] <= r[cl])
                            continue;
                          double d_assigned = 0.0;
#pragma omp simd reduction(+ : d_assigned) simdlen(4)
                          for (std::size_t j = 0; j < m; j++)
                          {
                            double diff = X[j * n + i] - centroids_[j * n_clusters_ + cl];
                            d_assigned += diff * diff;
                          }
                          d_assigned = std::sqrt(d_assigned);
                          u[i] = d_assigned;
                          for (int k = 0; k < n_clusters_; k++)
                          {
                            if (k == cl)
                              continue;
                            if (u[i] <= l[i * n_clusters_ + k])
                              continue;
                            double d = 0.0;
#pragma omp simd reduction(+ : d) simdlen(4)
                            for (std::size_t j = 0; j < m; j++)
                            {
                              double diff = X[j * n + i] - centroids_[j * n_clusters_ + k];
                              d += diff * diff;
                            }
                            d = std::sqrt(d);
                            l[i * n_clusters_ + k] = d;
                            if (d < u[i])
                            {
                              u[i] = d;
                              labels[i] = k;
                              any_changed.store(true, std::memory_order_relaxed);
                            }
                          }
                        }
                      });
    if (!any_changed.load(std::memory_order_relaxed))
      break; // 収束

    // ----------------------------
    // 下界の再計算: 各サンプルについて、所属クラスタ以外の最小距離を再計算して下界に設定
    // ----------------------------
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const tbb::blocked_range<size_t> &range)
                      {
                        for (size_t i = range.begin(); i < range.end(); i++)
                        {
                          int cl = labels[i];
                          double second_best = std::numeric_limits<double>::max();
                          for (int k = 0; k < n_clusters_; k++)
                          {
                            if (k == cl)
                              continue;
                            double d = 0.0;
#pragma omp simd reduction(+ : d) simdlen(4)
                            for (std::size_t j = 0; j < m; j++)
                            {
                              double diff = X[j * n + i] - centroids_[j * n_clusters_ + k];
                              d += diff * diff;
                            }
                            d = std::sqrt(d);
                            if (d < second_best)
                              second_best = d;
                          }
                          // 所属クラスタ以外の最小距離を下界として再設定（ここでは例として、所属クラスタの下界は u[i] にする）
                          l[i * n_clusters_ + cl] = u[i];
                        }
                      });
  } // end main iteration loop
}
// ----------------------------------------------------------------------
// run_hamerly: Hamerly’s Algorithm
// ----------------------------------------------------------------------
void KMeans::run_hamerly(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  // -------------------------------
  // 初期割り当て：各点について、最小距離 (u) と第二最小距離 (l) を計算
  // -------------------------------
  std::vector<int> labels(n, -1);
  std::vector<double> u(n, std::numeric_limits<double>::max());
  std::vector<double> l(n, 0.0);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                    [&](const tbb::blocked_range<size_t> &r)
                    {
                      for (size_t i = r.begin(); i != r.end(); i++)
                      {
                        double best = std::numeric_limits<double>::max();
                        double second_best = std::numeric_limits<double>::max();
                        int best_cl = -1;
                        for (int k = 0; k < n_clusters_; k++)
                        {
                          double dist_sq = 0.0;
                          for (std::size_t j = 0; j < m; j++)
                          {
                            double diff = X[j * n + i] - centroids_[j * n_clusters_ + k];
                            dist_sq += diff * diff;
                          }
                          if (dist_sq < best)
                          {
                            second_best = best;
                            best = dist_sq;
                            best_cl = k;
                          }
                          else if (dist_sq < second_best)
                          {
                            second_best = dist_sq;
                          }
                        }
                        labels[i] = best_cl;
                        u[i] = std::sqrt(best);
                        l[i] = std::sqrt(second_best);
                      }
                    });

  // 主反復ループ
  for (int iter = 0; iter < max_iter_; iter++)
  {
    // -------------------------------
    // 更新ステップ：各クラスタ中心の再計算
    // -------------------------------
    std::vector<double> new_centroids(n_clusters_ * m, 0.0);
    std::vector<int> counts(n_clusters_, 0);

    // TBB を用いた局所加算（各スレッドが局所バッファを持ち、後で統合）
    tbb::spin_mutex mutex;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const tbb::blocked_range<size_t> &r)
                      {
                        std::vector<double> local_sum(n_clusters_ * m, 0.0);
                        std::vector<int> local_count(n_clusters_, 0);
                        for (size_t i = r.begin(); i != r.end(); i++)
                        {
                          int cl = labels[i];
                          local_count[cl]++;
                          // 加算部分：BLAS の daxpy で代替可能（ここでは単純ループ）
                          for (std::size_t j = 0; j < m; j++)
                          {
                            local_sum[j * n_clusters_ + cl] += X[j * n + i];
                          }
                        }
                        // 結果をグローバルバッファに加算
                        tbb::spin_mutex::scoped_lock lock(mutex);
                        for (int k = 0; k < n_clusters_; k++)
                        {
                          counts[k] += local_count[k];
                          for (std::size_t j = 0; j < m; j++)
                          {
                            new_centroids[j * n_clusters_ + k] += local_sum[j * n_clusters_ + k];
                          }
                        }
                      });

    // 各クラスタごとに平均を取る
    for (int k = 0; k < n_clusters_; k++)
    {
      if (counts[k] > 0)
      {
        double inv = 1.0 / counts[k];
        // BLAS の scaling 関数 cblas_dscal を使ってもよい
        for (std::size_t j = 0; j < m; j++)
        {
          new_centroids[j * n_clusters_ + k] *= inv;
        }
      }
      else
      {
        // ランダム再初期化
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::size_t> dis(0, n - 1);
        std::size_t idx = dis(gen);
        for (std::size_t j = 0; j < m; j++)
        {
          new_centroids[j * n_clusters_ + k] = X[j * n + idx];
        }
      }
    }

    // 計算済みの新しい中心との移動量 shift[k]
    std::vector<double> shift(n_clusters_, 0.0);
    tbb::parallel_for(tbb::blocked_range<int>(0, n_clusters_),
                      [&](const tbb::blocked_range<int> &r)
                      {
                        for (int k = r.begin(); k < r.end(); k++)
                        {
                          double d = 0.0;
                          for (std::size_t j = 0; j < m; j++)
                          {
                            double diff = centroids_[j * n_clusters_ + k] - new_centroids[j * n_clusters_ + k];
                            d += diff * diff;
                          }
                          shift[k] = std::sqrt(d);
                        }
                      });

    // 更新前の中心と置き換え
    centroids_ = new_centroids;

    // 最大シフト量
    double max_shift = *std::max_element(shift.begin(), shift.end());

    // -------------------------------
    // 各クラスタについて安全域 s[k] の計算
    // s[k] = 0.5 * min_{l != k} distance(centroid_k, centroid_l)
    // -------------------------------
    std::vector<double> s(n_clusters_, std::numeric_limits<double>::max());
    tbb::parallel_for(tbb::blocked_range<int>(0, n_clusters_),
                      [&](const tbb::blocked_range<int> &r)
                      {
                        for (int k = r.begin(); k < r.end(); k++)
                        {
                          double min_dist = std::numeric_limits<double>::max();
                          for (int kk = 0; kk < n_clusters_; kk++)
                          {
                            if (k == kk)
                              continue;
                            double d = 0.0;
                            for (std::size_t j = 0; j < m; j++)
                            {
                              double diff = centroids_[j * n_clusters_ + k] - centroids_[j * n_clusters_ + kk];
                              d += diff * diff;
                            }
                            d = std::sqrt(d);
                            if (d < min_dist)
                              min_dist = d;
                          }
                          s[k] = 0.5 * min_dist;
                        }
                      });

    // -------------------------------
    // 境界の更新：上界 u[i] は所属クラスタの shift 分だけ増加、下界 l[i] は max_shift 分だけ減少
    // -------------------------------
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const tbb::blocked_range<size_t> &r)
                      {
                        for (size_t i = r.begin(); i != r.end(); i++)
                        {
                          int cl = labels[i];
                          u[i] += shift[cl];
                          l[i] = std::max(l[i] - max_shift, 0.0);
                        }
                      });

    // -------------------------------
    // 再割り当てステップ
    // -------------------------------
    // 変更があったかどうかのフラグ用
    std::atomic<bool> any_changed(false);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const tbb::blocked_range<size_t> &r)
                      {
                        for (size_t i = r.begin(); i != r.end(); i++)
                        {
                          int cl = labels[i];
                          // 安全域が十分ならスキップ
                          if (u[i] <= s[cl])
                            continue;
                          // 所属中心との距離を再計算
                          double d_assigned = 0.0;
                          for (std::size_t j = 0; j < m; j++)
                          {
                            double diff = X[j * n + i] - centroids_[j * n_clusters_ + cl];
                            d_assigned += diff * diff;
                          }
                          d_assigned = std::sqrt(d_assigned);
                          u[i] = d_assigned;
                          // 他の中心との比較
                          for (int k = 0; k < n_clusters_; k++)
                          {
                            if (k == cl)
                              continue;
                            // 安全条件：if (u[i] <= 0.5 * distance(centroid[cl], centroid[k])) continue;
                            double d_center = 0.0;
                            for (std::size_t j = 0; j < m; j++)
                            {
                              double diff = centroids_[j * n_clusters_ + cl] - centroids_[j * n_clusters_ + k];
                              d_center += diff * diff;
                            }
                            d_center = std::sqrt(d_center);
                            if (u[i] <= 0.5 * d_center)
                              continue;
                            double d = 0.0;
                            for (std::size_t j = 0; j < m; j++)
                            {
                              double diff = X[j * n + i] - centroids_[j * n_clusters_ + k];
                              d += diff * diff;
                            }
                            d = std::sqrt(d);
                            if (d < u[i])
                            {
                              u[i] = d;
                              labels[i] = k;
                              any_changed.store(true, std::memory_order_relaxed);
                            }
                          }
                        }
                      });

    // -------------------------------
    // 下界の再計算：各点について、所属クラスタ以外の最小距離を下界として設定
    // -------------------------------
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const tbb::blocked_range<size_t> &r)
                      {
                        for (size_t i = r.begin(); i != r.end(); i++)
                        {
                          int cl = labels[i];
                          double second_best = std::numeric_limits<double>::max();
                          for (int k = 0; k < n_clusters_; k++)
                          {
                            if (k == cl)
                              continue;
                            double d = 0.0;
                            for (std::size_t j = 0; j < m; j++)
                            {
                              double diff = X[j * n + i] - centroids_[j * n_clusters_ + k];
                              d += diff * diff;
                            }
                            d = std::sqrt(d);
                            if (d < second_best)
                              second_best = d;
                          }
                          l[i] = second_best;
                        }
                      });

    // 収束判定
    if (!any_changed.load(std::memory_order_relaxed))
      break;
  } // iter loop
}

// ----------------------------------------------------------------------
// fit: 学習 (アルゴリズムに応じて実行)
// ----------------------------------------------------------------------
void KMeans::fit(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  // k-means++ による初期化
  initialize_centroids(X, n, m);

  is_initialized_ = true;

  // サンプルごとのクラスタ割り当て（0-indexed）
  std::vector<int> labels(n, -1);
  std::vector<int> new_labels(n, -1);

  // 反復処理
  for (int iter = 0; iter < max_iter_; iter++)
  {
    bool assignment_changed = false;

    // ===== 割り当てステップ =====
    if (use_kdtree_)
    {
      // centroids_ は Col-Major形式 (n_clusters_ x m)
      // KDTreeはrow-major形式を前提としているため、一旦変換する
      std::vector<double> centroids_row(n_clusters_ * m);
      for (int k = 0; k < n_clusters_; k++)
      {
        for (std::size_t j = 0; j < m; j++)
        {
          // row-major: 各クラスタ中心 k の特徴量 j は contigously 配列の [k * m + j] に配置
          centroids_row[k * m + j] = centroids_[j * n_clusters_ + k];
        }
      }
      KDTree tree(centroids_row, static_cast<std::size_t>(n_clusters_), m);

      // 各サンプルについて、KDTreeを用いて最近傍クラスタを探索
      for (std::size_t i = 0; i < n; i++)
      {
        // サンプル i の m次元ベクトルを一時バッファにコピー（XはCol-Major）
        std::vector<double> sample(m);
        for (std::size_t j = 0; j < m; j++)
        {
          sample[j] = X[j * n + i];
        }
        double bestDist = std::numeric_limits<double>::max();
        int nearest = tree.nearestNeighbor(sample.data(), bestDist);
        new_labels[i] = nearest;
      }
    }
    else
    {
      // 全探索による割り当て
      for (std::size_t i = 0; i < n; i++)
      {
        int best_cluster = -1;
        double best_dist = std::numeric_limits<double>::max();
        for (int k = 0; k < n_clusters_; k++)
        {
          double d = 0.0;
          for (std::size_t j = 0; j < m; j++)
          {
            double diff = X[j * n + i] - centroids_[j * n_clusters_ + k];
            d += diff * diff;
          }
          if (d < best_dist)
          {
            best_dist = d;
            best_cluster = k;
          }
        }
        new_labels[i] = best_cluster;
      }
    }

    // 収束チェック：前回の割り当てと比較
    if (iter > 0)
    {
      for (std::size_t i = 0; i < n; i++)
      {
        if (new_labels[i] != labels[i])
        {
          assignment_changed = true;
          break;
        }
      }
      if (!assignment_changed)
      {
        break; // 割り当てが変わらなければ収束
      }
    }
    labels = new_labels;

    // ===== 更新ステップ =====
    std::vector<double> new_centroids(n_clusters_ * m, 0.0);
    std::vector<int> counts(n_clusters_, 0);

    // 各サンプルのデータを、所属クラスタに加算
    for (std::size_t i = 0; i < n; i++)
    {
      int cluster = labels[i];
      counts[cluster]++;
      for (std::size_t j = 0; j < m; j++)
      {
        new_centroids[j * n_clusters_ + cluster] += X[j * n + i];
      }
    }

    // 各クラスタごとに平均を計算
    for (int k = 0; k < n_clusters_; k++)
    {
      if (counts[k] > 0)
      {
        double inv = 1.0 / counts[k];
        for (std::size_t j = 0; j < m; j++)
        {
          new_centroids[j * n_clusters_ + k] *= inv;
        }
      }
      else
      {
        // サンプルが割り当てられていないクラスタはランダムなサンプルで再初期化
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::size_t> dis(0, n - 1);
        std::size_t rand_index = dis(gen);
        for (std::size_t j = 0; j < m; j++)
        {
          new_centroids[j * n_clusters_ + k] = X[j * n + rand_index];
        }
      }
    }

    // クラスタ中心のシフト量を計算
    double total_shift = 0.0;
    for (int k = 0; k < n_clusters_; k++)
    {
      double d = 0.0;
      for (std::size_t j = 0; j < m; j++)
      {
        double diff = centroids_[j * n_clusters_ + k] - new_centroids[j * n_clusters_ + k];
        d += diff * diff;
      }
      total_shift += std::sqrt(d);
    }
    centroids_ = new_centroids;

    if (total_shift < tol_)
    {
      break; // 収束
    }
  }
}

// ----------------------------------------------------------------------
// predict: 各サンプルに対して最も近いクラスタラベルを返す
// 出力は (n) 要素の配列（各要素はクラスタ番号を double に変換したもの）
// ----------------------------------------------------------------------
std::vector<double> KMeans::predict(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  std::vector<double> predictions(n, 0.0);

  if (use_kdtree_)
  {

    // centroids_ は Col-Major形式 (n_clusters_ x m)
    // KDTreeはrow-major形式を前提としているため、一旦変換する
    std::vector<double> centroids_row(n_clusters_ * m);
    for (int k = 0; k < n_clusters_; k++)
    {
      for (std::size_t j = 0; j < m; j++)
      {
        // row-major: 各クラスタ中心 k の特徴量 j は contigously 配列の [k * m + j] に配置
        centroids_row[k * m + j] = centroids_[j * n_clusters_ + k];
      }
    }
    KDTree tree(centroids_row, static_cast<std::size_t>(n_clusters_), m);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; i++)
    {
      // 各サンプルの特徴量は連続していないため、一時バッファにコピー
      std::vector<double> sample(m);
      for (std::size_t j = 0; j < m; j++)
      {
        sample[j] = X[j * n + i];
      }
      double bestDist;
      int idx = tree.nearestNeighbor(sample.data(), bestDist);
      predictions[i] = static_cast<double>(idx);
    }
  }
  else
  {
// 全探索による割り当て
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; i++)
    {
      int best_cluster = -1;
      double best_dist = std::numeric_limits<double>::max();
      for (int k = 0; k < n_clusters_; k++)
      {
        double d = 0.0;
        for (std::size_t j = 0; j < m; j++)
        {
          double diff = X[j * n + i] - centroids_[j * n_clusters_ + k];
          d += diff * diff;
        }
        if (d < best_dist)
        {
          best_dist = d;
          best_cluster = k;
        }
      }
      predictions[i] = static_cast<double>(best_cluster);
    }
  }
  return predictions;
}

// ----------------------------------------------------------------------
// transform: 各サンプルと各クラスタ中心間の距離を計算 (出力サイズ: n × n_clusters_)
// 出力は row-major として、インデックス (i, k) は [i * n_clusters_ + k] に格納
// ----------------------------------------------------------------------
std::vector<double> KMeans::transform(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  std::vector<double> output(n * n_clusters_, 0.0);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n; i++)
  {
    for (int k = 0; k < n_clusters_; k++)
    {
      double d = 0.0;
      for (std::size_t j = 0; j < m; j++)
      {
        double diff = X[j * n + i] - centroids_[j * n_clusters_ + k];
        d += diff * diff;
      }
      output[i * n_clusters_ + k] = std::sqrt(d);
    }
  }
  return output;
}

// ----------------------------------------------------------------------
// fit_transform: fit の後に transform を実行
// ----------------------------------------------------------------------
std::vector<double> KMeans::fit_transform(const std::vector<double> &X, std::size_t n, std::size_t m)
{
  fit(X, n, m);
  return transform(X, n, m);
}


