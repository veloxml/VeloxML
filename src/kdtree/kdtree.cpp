#include "kdtree/kdtree.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <cblas.h>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// コンストラクタ: 入力は Col-Major 形式の (n × m) 行列
KDTree::KDTree(const std::vector<double> &points, std::size_t n, std::size_t m)
    : dimensions_(m)
{
  std::vector<std::pair<int, const double *>> items(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    items[i] = {static_cast<int>(i), &points[i * m]};
  }
  root_ = buildTree(items, 0);
}

// KDTree の構築（再帰関数）
// ※ 引数 items を値渡ししているので、タスク呼び出し時にコピーは発生しません。
std::unique_ptr<KDTreeNode> KDTree::buildTree(
    std::vector<std::pair<int, const double *>> items, int depth)
{
  if (items.empty())
    return nullptr;

  int axis = depth % dimensions_;
  std::size_t median = items.size() / 2;

  // OpenMP 並列領域内で nth_element を実行
#pragma omp parallel
  {
#pragma omp single nowait
    {
      std::nth_element(items.begin(), items.begin() + median, items.end(),
                       [axis](const auto &a, const auto &b)
                       {
                         return a.second[axis] < b.second[axis];
                       });
    }
  }

  auto node = std::make_unique<KDTreeNode>();
  node->point = items[median].second;
  node->index = items[median].first;
  node->split_dim = axis;

  // タスク内での直接の unique_ptr のコピーはできないため、
  // ローカル変数にタスク結果を格納し、タスク終了後にムーブする。
  std::unique_ptr<KDTreeNode> left_subtree;
  std::unique_ptr<KDTreeNode> right_subtree;

#pragma omp task firstprivate(depth, median) shared(left_subtree, items)
  {
    left_subtree = buildTree(std::vector<std::pair<int, const double *>>(
                                   items.begin(), items.begin() + median),
                               depth + 1);
  }
#pragma omp task firstprivate(depth, median) shared(right_subtree, items)
  {
    right_subtree = buildTree(std::vector<std::pair<int, const double *>>(
                                   items.begin() + median + 1, items.end()),
                               depth + 1);
  }
#pragma omp taskwait

  node->left = std::move(left_subtree);
  node->right = std::move(right_subtree);

  return node;
}

// 最近傍探索
int KDTree::nearestNeighbor(const double *query, double &bestDist) const
{
  int bestIndex = -1;
  bestDist = std::numeric_limits<double>::max();
  searchNearest(root_.get(), query, bestIndex, bestDist);
  return bestIndex;
}

// 再帰的に最近傍を探索
void KDTree::searchNearest(
    const KDTreeNode *node,
    const double *query,
    int &bestIndex,
    double &bestDist) const
{
  if (!node)
    return;

  double dist = sq_euclidean_distance(node->point, query, dimensions_);
  if (dist < bestDist)
  {
    bestDist = dist;
    bestIndex = node->index;
  }

  // 探索すべき枝を決定
  int axis = node->split_dim;
  bool leftFirst = query[axis] < node->point[axis];

  searchNearest(leftFirst ? node->left.get() : node->right.get(), query, bestIndex, bestDist);
  if (std::abs(query[axis] - node->point[axis]) < bestDist)
  {
    searchNearest(leftFirst ? node->right.get() : node->left.get(), query, bestIndex, bestDist);
  }
}

// SIMD を用いたユークリッド距離の二乗計算
double KDTree::sq_euclidean_distance(const double *a, const double *b, std::size_t m)
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
