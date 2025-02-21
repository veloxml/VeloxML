#include "tree/random_forest_classification.hpp"
#include <algorithm>
#include <random>
#include <unordered_map>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

// コンストラクタ
RandomForestClassification::RandomForestClassification(
    int n_trees,
    int max_depth,
    int min_samples_leaf,
    int min_samples_split,
    double min_impurity_decrease,
    int max_leaf_nodes,
    int max_bins,
    Criterion tree_mode,
    SplitAlgorithm tree_split_mode,
    int max_features,
    int n_jobs,
    int random_seed)
    : n_trees_(n_trees), max_depth_(max_depth), min_samples_leaf_(min_samples_leaf),
      min_samples_split_(min_samples_split), min_impurity_decrease_(min_impurity_decrease), max_leaf_nodes_(max_leaf_nodes),
      max_bins_(max_bins), max_features_(max_features), tree_mode_(tree_mode),
      tree_split_mode_(tree_split_mode), n_jobs_(n_jobs), random_seed_(random_seed), is_initialized_(false)
{
  if (random_seed_ == -1){
    rng_.seed(std::random_device{}());
  }else{
    rng_.seed(random_seed);
  }
}

RandomForestClassification::~RandomForestClassification() {}

//
// fit: Xは n×m の col-major 行列, Yは n 要素のラベル
//
void RandomForestClassification::fit(const std::vector<double> &X,
                                     const std::vector<double> &Y,
                                     std::size_t n, std::size_t m)
{
  num_features_ = static_cast<int>(m);

  // ツリーの初期化
  trees_.clear();
  trees_.resize(n_trees_);

  is_initialized_ = true;

  // 各木用のブートストラップサンプルのインデックス生成（シリアル生成）
  std::uniform_int_distribution<int> dist(0, static_cast<int>(n) - 1);
  std::vector<std::vector<int>> bootstrap_indices(n_trees_, std::vector<int>(n));
  for (int t = 0; t < n_trees_; t++)
  {
    for (std::size_t i = 0; i < n; i++)
    {
      bootstrap_indices[t][i] = dist(rng_);
    }
  }

  // 各決定木の学習は、TBBのparallel_forを用いて木レベルで並列化
  tbb::parallel_for(0, n_trees_, [&, n, m](int t)
                    {
        trees_[t] = std::make_unique<DecisionTreeClassification>(
            max_depth_, min_samples_split_, max_bins_, tree_mode_, tree_split_mode_,
            min_samples_leaf_,  max_leaf_nodes_,
            min_impurity_decrease_, max_features_);
        fit_tree(X, Y, n, m, t, bootstrap_indices[t]); });
}

//
// fit_tree: 各決定木の学習処理（ブートストラップサンプルを抽出）
// X, Yはすでにcol-major形式の1次元配列
//
void RandomForestClassification::fit_tree(const std::vector<double> &X,
                                          const std::vector<double> &Y,
                                          std::size_t n, std::size_t m,
                                          int tree_index,
                                          const std::vector<int> &bootstrap_indices)
{
  // ブートストラップサンプル用のX_boot, Y_bootを作成
  std::vector<double> X_boot(n * m);
  std::vector<double> Y_boot(n);
  // ここでは、各サンプルiに対して、Xのcol-majorアクセス: 各特徴量 j は X[j*n + idx]
  for (std::size_t i = 0; i < n; i++)
  {
    int idx = bootstrap_indices[i];
    for (std::size_t j = 0; j < m; j++)
    {
      X_boot[j * n + i] = X[j * n + idx];
    }
    Y_boot[i] = Y[idx];
  }
  trees_[tree_index]->fit(X_boot, Y_boot, n, m);
}

//
// predict: 各決定木の予測結果の多数決を行う
//
std::vector<double> RandomForestClassification::predict(const std::vector<double> &X,
                                                        std::size_t l, std::size_t m)
{
  // 各決定木の予測結果をTBBで並列取得
  std::vector<std::vector<double>> all_preds(n_trees_);
  tbb::parallel_for(0, n_trees_, [&, l, m](int t)
                    { all_preds[t] = trees_[t]->predict(X, l, m); });

  // 多数決による最終予測
  std::vector<double> final_preds(l, 0);
  for (std::size_t i = 0; i < l; i++)
  {
    std::unordered_map<int, int> votes;
    for (int t = 0; t < n_trees_; t++)
    {
      int pred = static_cast<int>(all_preds[t][i]);
      votes[pred]++;
    }
    int best_label = 0;
    int best_count = 0;
    for (const auto &kv : votes)
    {
      if (kv.second > best_count)
      {
        best_count = kv.second;
        best_label = kv.first;
      }
    }
    final_preds[i] = best_label;
  }
  return final_preds;
}

//
// predict_proba: 各決定木の確率予測の平均を返す
//
std::vector<double> RandomForestClassification::predict_proba(const std::vector<double> &X,
                                                              std::size_t l, std::size_t m)
{
  std::vector<std::vector<double>> all_probas(n_trees_);
  tbb::parallel_for(0, n_trees_, [&, l, m](int t)
                    { all_probas[t] = trees_[t]->predict_proba(X, l, m); });

  int num_classes = 1;
  if (!all_probas.empty() && !all_probas[0].empty())
    num_classes = static_cast<int>(all_probas[0].size()) / static_cast<int>(l);

  std::vector<double> final_probas(l * num_classes, 0.0);
  // 各サンプルごとに、各木の予測確率を平均
  for (std::size_t i = 0; i < l; i++)
  {
    for (int t = 0; t < n_trees_; t++)
    {
      for (int c = 0; c < num_classes; c++)
      {
        // 各木の出力はcol-major形式: [c*l + i]
        final_probas[c * l + i] += all_probas[t][c * l + i];
      }
    }
    for (int c = 0; c < num_classes; c++)
    {
      final_probas[c * l + i] /= n_trees_;
    }
  }
  return final_probas;
}

//
// feature_importances: 各決定木の特徴量重要度の平均を計算
//
std::vector<double> RandomForestClassification::feature_importances() const
{
  std::vector<double> total_importance(num_features_, 0.0);
  for (const auto &tree : trees_)
  {
    std::vector<double> tree_importance = tree->compute_feature_importance();
    for (int i = 0; i < num_features_; i++)
    {
      total_importance[i] += tree_importance[i];
    }
  }
  for (int i = 0; i < num_features_; i++)
  {
    total_importance[i] /= trees_.size();
  }
  return total_importance;
}

//
// Getter implementations
//
int RandomForestClassification::get_n_trees() const { return n_trees_; }
int RandomForestClassification::get_max_depth() const { return max_depth_; }
int RandomForestClassification::get_min_samples_leaf() const { return min_samples_leaf_; }
int RandomForestClassification::get_min_samples_split() const { return min_samples_split_; }
double RandomForestClassification::get_min_impurity_decrease() const { return min_impurity_decrease_; }
int RandomForestClassification::get_max_bins() const { return max_bins_; }
int RandomForestClassification::get_max_features() const { return max_features_; }
Criterion RandomForestClassification::get_tree_mode() const { return tree_mode_; }
SplitAlgorithm RandomForestClassification::get_tree_split_mode() const { return tree_split_mode_; }
int RandomForestClassification::get_n_jobs() const { return n_jobs_; }
