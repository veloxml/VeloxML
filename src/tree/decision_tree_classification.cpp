#include "tree/decision_tree_classification.hpp"
#include <limits>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <random>

// 並列処理・SIMD用ライブラリ
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>
#include <omp.h>

//
// コンストラクタ／デストラクタ
//
DecisionTreeClassification::DecisionTreeClassification(int max_depth, int min_samples_split, int max_bins,
                                                       Criterion criterion, SplitAlgorithm split_algorithm,
                                                       int min_samples_leaf, int max_leaf_nodes,
                                                       double min_impurity_decrease, int max_features)
    : max_depth_(max_depth), min_samples_split_(min_samples_split), max_bins_(max_bins),
      criterion_(criterion), split_algorithm_(split_algorithm),
      min_samples_leaf_(min_samples_leaf), max_leaf_nodes_(max_leaf_nodes),
      min_impurity_decrease_(min_impurity_decrease), max_features_(max_features)
{
  // 初期化: 現在の葉ノード数を 0 に設定
  current_leaf_count_ = 0;
}

DecisionTreeClassification::~DecisionTreeClassification() { }

//
// fit: Xは n×m の col-major 行列, Yは n 要素のラベル
//
void DecisionTreeClassification::fit(const std::vector<double> &X,
                                     const std::vector<double> &Y,
                                     std::size_t n, std::size_t m)
{
  // 保存しておく特徴量数
  num_features_ = static_cast<int>(m);
  current_leaf_count_ = 0;

  tree_.clear();
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  build_tree(X, n, m, Y, indices, 0);
}

//
// predict: l×m の特徴行列 X を入力し、各サンプルの予測クラスを返す
//
std::vector<double> DecisionTreeClassification::predict(const std::vector<double> &X,
                                                        std::size_t l, std::size_t m)
{
  std::vector<double> predictions(l, 0);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < l; i++)
  {
    int node_index = 0;
    while (!tree_[node_index].is_leaf)
    {
      int feature = tree_[node_index].feature_index;
      // col-major: サンプル i の値は X[feature * l + i]
      double value = X[feature * l + i];
      node_index = (value <= tree_[node_index].threshold)
                       ? tree_[node_index].left
                       : tree_[node_index].right;
    }
    predictions[i] = tree_[node_index].predicted_class;
  }
  return predictions;
}

//
// predict_proba: 各サンプルのクラス確率（出力は 1 次元 vector, サイズ = l×c）を返す
//
std::vector<double> DecisionTreeClassification::predict_proba(const std::vector<double> &X,
                                                              std::size_t l, std::size_t m)
{
  int num_classes = 1;
  if (!tree_.empty() && !tree_[0].probabilities.empty())
    num_classes = static_cast<int>(tree_[0].probabilities.size());
  std::vector<double> result(l * num_classes, 0.0);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < l; i++)
  {
    int node_index = 0;
    while (!tree_[node_index].is_leaf)
    {
      int feature = tree_[node_index].feature_index;
      double value = X[feature * l + i];
      node_index = (value <= tree_[node_index].threshold)
                       ? tree_[node_index].left
                       : tree_[node_index].right;
    }
    const std::vector<double> &probs = tree_[node_index].probabilities;
    for (int j = 0; j < num_classes; j++)
      result[i * num_classes + j] = probs[j];
  }
  return result;
}

//
// CandidateImpurity: 各候補ビンでの不純度値とそのビン番号を保持
//
struct CandidateImpurity
{
  double impurity;
  int bin;
};

inline CandidateImpurity min_candidate(const CandidateImpurity &a, const CandidateImpurity &b)
{
  return (a.impurity < b.impurity) ? a : b;
}

//
// find_best_threshold:
// ヒストグラム・累積ヒストグラム領域を連続メモリで確保し、
// メモリアクセスの局所性を向上させながら、各候補ビンの不純度をTBB/OMPで計算
//
void DecisionTreeClassification::find_best_threshold(
    const std::vector<double> &X, std::size_t n, std::size_t m,
    const std::vector<double> &Y, const std::vector<int> &indices,
    int feature_index, int num_classes,
    double &best_threshold, double &best_impurity,
    std::vector<int> &best_left_indices, std::vector<int> &best_right_indices) const
{
  // 1. 最小値・最大値取得
  double min_val = std::numeric_limits<double>::max();
  double max_val = std::numeric_limits<double>::lowest();
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
  for (std::size_t i = 0; i < indices.size(); i++)
  {
    double value = X[feature_index * n + indices[i]];
    if (value < min_val)
      min_val = value;
    if (value > max_val)
      max_val = value;
  }
  if (min_val == max_val)
  {
    best_threshold = min_val;
    best_impurity = 0.0;
    best_left_indices = indices;
    best_right_indices.clear();
    return;
  }

  // 2. ヒストグラム領域（1次元配列）確保
  const int num_bins = max_bins_;
  double bin_width = (max_val - min_val) / num_bins;
  std::vector<int> hist(num_bins * num_classes, 0);

// 3. ヒストグラム集計
#pragma omp parallel
  {
    std::vector<int> local_hist(num_bins * num_classes, 0);
#pragma omp for nowait
    for (std::size_t i = 0; i < indices.size(); i++)
    {
      std::size_t idx = indices[i];
      double value = X[feature_index * n + idx];
      int bin = static_cast<int>((value - min_val) / bin_width);
      if (bin >= num_bins)
        bin = num_bins - 1;
      int label = static_cast<int>(Y[idx]);
      if (label >= 0 && label < num_classes)
        local_hist[bin * num_classes + label]++;
    }
#pragma omp critical
    {
      for (int b = 0; b < num_bins; b++)
      {
        for (int c = 0; c < num_classes; c++)
          hist[b * num_classes + c] += local_hist[b * num_classes + c];
      }
    }
  }

  // 4. 全体クラス分布
  std::vector<int> total_counts(num_classes, 0);
  int total_samples = static_cast<int>(indices.size());
  for (int b = 0; b < num_bins; b++)
  {
    for (int c = 0; c < num_classes; c++)
      total_counts[c] += hist[b * num_classes + c];
  }

  // 5. 累積ヒストグラム作成
  std::vector<int> cum_hist(num_bins * num_classes, 0);
  for (int c = 0; c < num_classes; c++)
    cum_hist[c] = hist[c];
  for (int b = 1; b < num_bins; b++)
  {
    for (int c = 0; c < num_classes; c++)
    {
      cum_hist[b * num_classes + c] = cum_hist[(b - 1) * num_classes + c] + hist[b * num_classes + c];
    }
  }

  // 6. TBBによる候補ビンごとの不純度計算
  CandidateImpurity bestCandidate = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, num_bins - 1),
      CandidateImpurity{std::numeric_limits<double>::max(), -1},
      [&](const tbb::blocked_range<int> &r, CandidateImpurity init) -> CandidateImpurity
      {
        for (int b = r.begin(); b < r.end(); b++)
        {
          int left_total = 0;
          for (int c = 0; c < num_classes; c++)
            left_total += cum_hist[b * num_classes + c];
          if (left_total == 0 || left_total == total_samples)
            continue;
          int right_total = total_samples - left_total;
          double impurity_candidate = 0.0;
          if (criterion_ == Criterion::Gini)
          {
            double left_sum = 0.0;
#pragma omp simd reduction(+ : left_sum)
            for (int c = 0; c < num_classes; c++)
            {
              double p = static_cast<double>(cum_hist[b * num_classes + c]) / left_total;
              left_sum += p * p;
            }
            double left_gini = 1.0 - left_sum;
            double right_sum = 0.0;
#pragma omp simd reduction(+ : right_sum)
            for (int c = 0; c < num_classes; c++)
            {
              int right_count = total_counts[c] - cum_hist[b * num_classes + c];
              double p = static_cast<double>(right_count) / right_total;
              right_sum += p * p;
            }
            double right_gini = 1.0 - right_sum;
            impurity_candidate = (left_total * left_gini + right_total * right_gini) / total_samples;
          }
          else if (criterion_ == Criterion::Entropy)
          {
            double left_entropy = 0.0;
#pragma omp simd reduction(+ : left_entropy)
            for (int c = 0; c < num_classes; c++)
            {
              double p = static_cast<double>(cum_hist[b * num_classes + c]) / left_total;
              if (p > 0)
                left_entropy -= p * std::log(p);
            }
            double right_entropy = 0.0;
#pragma omp simd reduction(+ : right_entropy)
            for (int c = 0; c < num_classes; c++)
            {
              int right_count = total_counts[c] - cum_hist[b * num_classes + c];
              double p = static_cast<double>(right_count) / right_total;
              if (p > 0)
                right_entropy -= p * std::log(p);
            }
            impurity_candidate = (left_total * left_entropy + right_total * right_entropy) / total_samples;
          }
          else if (criterion_ == Criterion::Logloss)
          {
            double left_logloss = 0.0;
#pragma omp simd reduction(+ : left_logloss)
            for (int c = 0; c < num_classes; c++)
            {
              if (cum_hist[b * num_classes + c] > 0)
              {
                double p = static_cast<double>(cum_hist[b * num_classes + c]) / left_total;
                left_logloss -= std::log(p);
              }
            }
            double right_logloss = 0.0;
#pragma omp simd reduction(+ : right_logloss)
            for (int c = 0; c < num_classes; c++)
            {
              int right_count = total_counts[c] - cum_hist[b * num_classes + c];
              if (right_count > 0)
              {
                double p = static_cast<double>(right_count) / right_total;
                right_logloss -= std::log(p);
              }
            }
            impurity_candidate = (left_total * left_logloss + right_total * right_logloss) / total_samples;
          }
          else
          {
            throw std::runtime_error("Unknown criterion in impurity calculation");
          }
          if (impurity_candidate < init.impurity)
          {
            init.impurity = impurity_candidate;
            init.bin = b;
          }
        }
        return init;
      },
      [](const CandidateImpurity &a, const CandidateImpurity &b) -> CandidateImpurity
      {
        return min_candidate(a, b);
      });

  // 7. フォールバック処理
  if (bestCandidate.bin < 0)
  {
    best_threshold = min_val + 0.5 * (max_val - min_val);
    best_left_indices = indices;
    best_right_indices.clear();
    best_impurity = 0.0;
    return;
  }

  // 8. 最適閾値の設定
  best_threshold = min_val + (bestCandidate.bin + 0.5) * bin_width;
  best_impurity = bestCandidate.impurity;

  // 9. 最適閾値による分割（シリアルスキャン）
  for (const auto idx : indices)
  {
    double value = X[feature_index * n + idx];
    if (value <= best_threshold)
      best_left_indices.push_back(idx);
    else
      best_right_indices.push_back(idx);
  }
}

//
// build_tree: ツリーを再帰的に構築
// 追加のハイパーパラメータ:
//   - min_samples_leaf: 葉に必要な最小サンプル数
//   - max_leaf_nodes: 最大葉ノード数（current_leaf_count_ で管理）
//   - min_impurity_decrease: 分割時の不純度低下がこの値未満なら分割しない
//   - max_features: 分割候補とする特徴量数（max_features_ < m の場合、ランダムサンプリング）
//
int DecisionTreeClassification::build_tree(const std::vector<double> &X, std::size_t n, std::size_t m,
                                           const std::vector<double> &Y, const std::vector<int> &indices, int depth)
{
  int node_index = static_cast<int>(tree_.size());
  tree_.push_back(Node());

  // ノードに到達したサンプル数を保持
  tree_[node_index].sample_count = static_cast<int>(indices.size());
  int num_classes = get_num_classes(Y);
  tree_[node_index].impurity = compute_impurity(indices, Y, num_classes);

  // 終了条件：最大深度、min_samples_split、または葉に必要な最小サンプル数に満たない場合
  if (depth >= max_depth_ || indices.size() < static_cast<std::size_t>(min_samples_split_) ||
      indices.size() < static_cast<std::size_t>(min_samples_leaf_))
  {
    tree_[node_index].is_leaf = true;
    current_leaf_count_++;
    std::vector<int> counts(num_classes, 0);
    for (const auto idx : indices)
    {
      int label = static_cast<int>(Y[idx]);
      if (label >= 0 && label < num_classes)
        counts[label]++;
    }
    int best_class = static_cast<int>(std::distance(counts.begin(), std::max_element(counts.begin(), counts.end())));
    tree_[node_index].predicted_class = best_class;
    tree_[node_index].probabilities.resize(num_classes, 0.0);
    for (int i = 0; i < num_classes; i++)
      tree_[node_index].probabilities[i] = static_cast<double>(counts[i]) / indices.size();
    return node_index;
  }

  // max_leaf_nodes の制約：もし既に最大葉数に達しているなら、現在のノードを葉にする
  if (current_leaf_count_ >= max_leaf_nodes_)
  {
    tree_[node_index].is_leaf = true;
    std::vector<int> counts(num_classes, 0);
    for (const auto idx : indices)
    {
      int label = static_cast<int>(Y[idx]);
      if (label >= 0 && label < num_classes)
        counts[label]++;
    }
    int best_class = static_cast<int>(std::distance(counts.begin(), std::max_element(counts.begin(), counts.end())));
    tree_[node_index].predicted_class = best_class;
    tree_[node_index].probabilities.resize(num_classes, 0.0);
    for (int i = 0; i < num_classes; i++)
      tree_[node_index].probabilities[i] = static_cast<double>(counts[i]) / indices.size();
    return node_index;
  }

  // max_features の適用: 分割候補とする特徴量集合をランダムサンプリング
  std::vector<int> feature_candidates;
  int num_feat = static_cast<int>(m);
  if (max_features_ > 0 && max_features_ < num_feat)
  {
    feature_candidates.resize(num_feat);
    std::iota(feature_candidates.begin(), feature_candidates.end(), 0);
    // シャッフル（乱数シードの管理は省略）
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(feature_candidates.begin(), feature_candidates.end(), g);
    feature_candidates.resize(max_features_);
  }
  else
  {
    // 全特徴量を候補とする
    feature_candidates.resize(num_feat);
    std::iota(feature_candidates.begin(), feature_candidates.end(), 0);
  }

  // 全候補特徴量について最適分割を探索
  double best_impurity_overall = std::numeric_limits<double>::max();
  int best_feature = -1;
  double best_thresh = 0.0;
  std::vector<int> best_left_indices;
  std::vector<int> best_right_indices;

  for (int f : feature_candidates)
  {
    double threshold_candidate = 0.0;
    double impurity_candidate = 0.0;
    std::vector<int> left_indices_candidate;
    std::vector<int> right_indices_candidate;

    find_best_threshold(X, n, m, Y, indices, f, num_classes,
                        threshold_candidate, impurity_candidate,
                        left_indices_candidate, right_indices_candidate);

    // 分割後の各葉が min_samples_leaf を満たしているかチェック
    if (left_indices_candidate.size() < static_cast<std::size_t>(min_samples_leaf_) ||
        right_indices_candidate.size() < static_cast<std::size_t>(min_samples_leaf_))
      continue;

    // 分割による不純度低下が min_impurity_decrease 未満の場合は無視
    double impurity_decrease = tree_[node_index].impurity - impurity_candidate;
    if (impurity_decrease < min_impurity_decrease_)
      continue;

    if (impurity_candidate < best_impurity_overall)
    {
      best_impurity_overall = impurity_candidate;
      best_feature = f;
      best_thresh = threshold_candidate;
      best_left_indices = left_indices_candidate;
      best_right_indices = right_indices_candidate;
    }
  }

  // 分割できなかった場合は葉ノードにする
  if (best_feature == -1)
  {
    tree_[node_index].is_leaf = true;
    current_leaf_count_++;
    std::vector<int> counts(num_classes, 0);
    for (const auto idx : indices)
    {
      int label = static_cast<int>(Y[idx]);
      if (label >= 0 && label < num_classes)
        counts[label]++;
    }
    int best_class = static_cast<int>(std::distance(counts.begin(), std::max_element(counts.begin(), counts.end())));
    tree_[node_index].predicted_class = best_class;
    tree_[node_index].probabilities.resize(num_classes, 0.0);
    for (int i = 0; i < num_classes; i++)
      tree_[node_index].probabilities[i] = static_cast<double>(counts[i]) / indices.size();
    return node_index;
  }

  // 内部ノードとして設定
  tree_[node_index].is_leaf = false;
  tree_[node_index].feature_index = best_feature;
  tree_[node_index].threshold = best_thresh;

  // 再帰的に左右サブツリーを構築
  int left_node = build_tree(X, n, m, Y, best_left_indices, depth + 1);
  int right_node = build_tree(X, n, m, Y, best_right_indices, depth + 1);
  tree_[node_index].left = left_node;
  tree_[node_index].right = right_node;

  return node_index;
}

//
// split_data: 標準分割（閾値との単純比較）
//
void DecisionTreeClassification::split_data(const std::vector<double> &X, std::size_t n, std::size_t m,
                                            const std::vector<int> &indices, int feature_index, double threshold,
                                            std::vector<int> &left_indices, std::vector<int> &right_indices) const
{
  for (const auto idx : indices)
  {
    double value = X[feature_index * n + idx];
    if (value <= threshold)
      left_indices.push_back(idx);
    else
      right_indices.push_back(idx);
  }
}

//
// split_data_histogram: ヒストグラム近似分割（SIMD最適化＋TBB並列化）
//
void DecisionTreeClassification::split_data_histogram(const std::vector<double> &X, std::size_t n, std::size_t m,
                                                      const std::vector<int> &indices, int feature_index, double threshold,
                                                      std::vector<int> &left_indices, std::vector<int> &right_indices) const
{
  double min_val = std::numeric_limits<double>::max();
  double max_val = std::numeric_limits<double>::lowest();
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
  for (std::size_t i = 0; i < indices.size(); i++)
  {
    double value = X[feature_index * n + indices[i]];
    if (value < min_val)
      min_val = value;
    if (value > max_val)
      max_val = value;
  }
  if (min_val == max_val)
  {
    left_indices = indices;
    return;
  }
  const int num_bins = max_bins_;
  double bin_width = (max_val - min_val) / num_bins;
  std::vector<int> bin_assignment(indices.size());
#pragma omp simd
  for (std::size_t i = 0; i < indices.size(); i++)
  {
    double value = X[feature_index * n + indices[i]];
    int bin = static_cast<int>((value - min_val) / bin_width);
    if (bin >= num_bins)
      bin = num_bins - 1;
    bin_assignment[i] = bin;
  }
  std::vector<tbb::concurrent_vector<int>> bins(num_bins);
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, indices.size()),
                    [&](const tbb::blocked_range<std::size_t> &r)
                    {
                      for (std::size_t i = r.begin(); i < r.end(); i++)
                      {
                        int bin = bin_assignment[i];
                        bins[bin].push_back(indices[i]);
                      }
                    });
  int threshold_bin = static_cast<int>((threshold - min_val) / bin_width);
  if (threshold_bin < 0)
    threshold_bin = 0;
  if (threshold_bin >= num_bins)
    threshold_bin = num_bins - 1;
  tbb::concurrent_vector<int> left_concurrent;
  tbb::concurrent_vector<int> right_concurrent;
  tbb::parallel_for(0, num_bins, [&](int b)
                    {
  if (b < threshold_bin) {
    for (int idx : bins[b])
      left_concurrent.push_back(idx);
  } else if (b > threshold_bin) {
    for (int idx : bins[b])
      right_concurrent.push_back(idx);
  } else {
    for (int idx : bins[b]) {
      double value = X[feature_index * n + idx];
      if (value <= threshold)
        left_concurrent.push_back(idx);
      else
        right_concurrent.push_back(idx);
    }
  } });
  left_indices.assign(left_concurrent.begin(), left_concurrent.end());
  right_indices.assign(right_concurrent.begin(), right_concurrent.end());
}

//
// 不純度計算（各指標に応じた計算）
//
double DecisionTreeClassification::compute_impurity(const std::vector<int> &indices, const std::vector<double> &Y,
                                                    int num_classes) const
{
  switch (criterion_)
  {
  case Criterion::Entropy:
    return compute_impurity_entropy(indices, Y, num_classes);
  case Criterion::Gini:
    return compute_impurity_gini(indices, Y, num_classes);
  case Criterion::Logloss:
    return compute_impurity_logloss(indices, Y, num_classes);
  default:
    throw std::runtime_error("Unknown criterion");
  }
}

double DecisionTreeClassification::compute_impurity_entropy(const std::vector<int> &indices,
                                                            const std::vector<double> &Y,
                                                            int num_classes) const
{
  std::size_t total = indices.size();
  std::vector<int> counts(num_classes, 0);
#pragma omp parallel
  {
    std::vector<int> local_counts(num_classes, 0);
#pragma omp for nowait
    for (std::size_t i = 0; i < total; i++)
    {
      int idx = indices[i];
      int label = static_cast<int>(Y[idx]);
      if (label >= 0 && label < num_classes)
        local_counts[label]++;
    }
#pragma omp critical
    {
      for (int c = 0; c < num_classes; c++)
        counts[c] += local_counts[c];
    }
  }
  double impurity = 0.0;
#pragma omp simd reduction(+ : impurity)
  for (int c = 0; c < num_classes; c++)
  {
    if (counts[c] > 0)
    {
      double p = static_cast<double>(counts[c]) / total;
      impurity -= p * std::log(p);
    }
  }
  return impurity;
}

double DecisionTreeClassification::compute_impurity_gini(const std::vector<int> &indices,
                                                         const std::vector<double> &Y,
                                                         int num_classes) const
{
  std::size_t total = indices.size();
  std::vector<int> counts(num_classes, 0);
#pragma omp parallel
  {
    std::vector<int> local_counts(num_classes, 0);
#pragma omp for nowait
    for (std::size_t i = 0; i < total; i++)
    {
      int idx = indices[i];
      int label = static_cast<int>(Y[idx]);
      if (label >= 0 && label < num_classes)
        local_counts[label]++;
    }
#pragma omp critical
    {
      for (int c = 0; c < num_classes; c++)
        counts[c] += local_counts[c];
    }
  }
  double sum_sq = 0.0;
#pragma omp simd reduction(+ : sum_sq)
  for (int c = 0; c < num_classes; c++)
  {
    double p = static_cast<double>(counts[c]) / total;
    sum_sq += p * p;
  }
  double impurity = 1.0 - sum_sq;
  return impurity;
}

double DecisionTreeClassification::compute_impurity_logloss(const std::vector<int> &indices,
                                                            const std::vector<double> &Y,
                                                            int num_classes) const
{
  std::size_t total = indices.size();
  std::vector<int> counts(num_classes, 0);
#pragma omp parallel
  {
    std::vector<int> local_counts(num_classes, 0);
#pragma omp for nowait
    for (std::size_t i = 0; i < total; i++)
    {
      int idx = indices[i];
      int label = static_cast<int>(Y[idx]);
      if (label >= 0 && label < num_classes)
        local_counts[label]++;
    }
#pragma omp critical
    {
      for (int c = 0; c < num_classes; c++)
        counts[c] += local_counts[c];
    }
  }
  double impurity = 0.0;
#pragma omp simd reduction(+ : impurity)
  for (int c = 0; c < num_classes; c++)
  {
    if (counts[c] > 0)
    {
      double p = static_cast<double>(counts[c]) / total;
      impurity -= std::log(p);
    }
  }
  return impurity;
}

//
// get_num_classes: Yに含まれる最大のラベル値+1を返す
//
int DecisionTreeClassification::get_num_classes(const std::vector<double> &Y) const
{
  int max_label = 0;
  for (double label : Y)
  {
    int ilabel = static_cast<int>(label);
    if (ilabel > max_label)
      max_label = ilabel;
  }
  return max_label + 1;
}

//
// accumulate_importance: ツリー全体を再帰走査して特徴量の重要度を集計する
//
void DecisionTreeClassification::accumulate_importance(int node_index, std::vector<double> &importances) const
{
  const Node &node = tree_[node_index];
  if (node.is_leaf)
    return;

  int left_index = node.left;
  int right_index = node.right;
  const Node &left_node = tree_[left_index];
  const Node &right_node = tree_[right_index];

  double weighted_child_impurity = (left_node.sample_count * left_node.impurity +
                                    right_node.sample_count * right_node.impurity) /
                                   static_cast<double>(node.sample_count);
  double reduction = node.impurity - weighted_child_impurity;
  importances[node.feature_index] += reduction * node.sample_count;

  accumulate_importance(left_index, importances);
  accumulate_importance(right_index, importances);
}

//
// compute_feature_importance: 全特徴量の重要度を正規化して返す
//
std::vector<double> DecisionTreeClassification::compute_feature_importance() const
{
  std::vector<double> importances(num_features_, 0.0);
  if (tree_.empty())
    return importances;

  accumulate_importance(0, importances);

  double total = 0.0;
  for (double v : importances)
    total += v;
  if (total > 0)
  {
    for (double &v : importances)
      v /= total;
  }
  return importances;
}