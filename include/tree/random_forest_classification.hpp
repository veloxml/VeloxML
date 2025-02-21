#ifndef RANDOM_FOREST_CLASSIFICATION_HPP
#define RANDOM_FOREST_CLASSIFICATION_HPP

#include <random>
#include <vector>
#include <memory>
#include "base/classification_base.hpp"
#include "tree/decision_tree_classification.hpp"

/**
 * @class RandomForestClassification
 * @brief
 * \if Japanese
 * ランダムフォレスト分類器
 *
 * 複数の決定木を学習し、分類タスクを実行するアンサンブル学習モデル。
 * 各決定木はブートストラップサンプルを用いて学習され、最終的な予測は多数決（または確率平均）によって決定される。
 * \else
 * Random Forest Classifier
 *
 * An ensemble learning model that trains multiple decision trees and performs classification tasks.
 * Each decision tree is trained on a bootstrap sample, and the final prediction is determined
 * by majority voting (or probability averaging).
 * \endif
 */
class RandomForestClassification : public ClassificationBase
{
public:
  /**
   * @brief
   * \if Japanese
   * ランダムフォレスト分類器のコンストラクタ
   * \else
   * Constructor for Random Forest Classifier
   * \endif
   *
   * @param n_trees
   * \if Japanese 決定木の本数 \else Number of decision trees \endif
   *
   * @param max_depth
   * \if Japanese 各決定木の最大深さ \else Maximum depth of each decision tree \endif
   *
   * @param min_samples_leaf
   * \if Japanese 葉ノードに必要な最小サンプル数 \else Minimum number of samples required in a leaf node \endif
   *
   * @param min_samples_split
   * \if Japanese 内部分割に必要な最小サンプル数 \else Minimum number of samples required to split a node \endif
   *
   * @param min_impurity_decrease
   * \if Japanese 分割による最小不純度減少量 \else Minimum impurity decrease required for a split \endif
   *
   * @param max_leaf_nodes
   * \if Japanese 最大葉ノード数 \else Maximum number of leaf nodes \endif
   *
   * @param max_bins
   * \if Japanese 最大ビン数（ヒストグラム分割用） \else Maximum number of bins (for histogram-based splitting) \endif
   *
   * @param tree_mode
   * \if Japanese 決定木の不純度評価基準 \else Criterion for decision tree impurity evaluation \endif
   *
   * @param tree_split_mode
   * \if Japanese 決定木の分割アルゴリズム \else Split algorithm for decision trees \endif
   *
   * @param max_features
   * \if Japanese 各分割で考慮する最大特徴量数 \else Maximum number of features considered per split \endif
   *
   * @param n_jobs
   * \if Japanese 並列処理で使用するスレッド数（デフォルト: 1） \else Number of threads used for parallel processing (default: 1) \endif
   */
  RandomForestClassification(int n_trees,
                             int max_depth,
                             int min_samples_leaf,
                             int min_samples_split,
                             double min_impurity_decrease,
                             int max_leaf_nodes,
                             int max_bins,
                             Criterion tree_mode,
                             SplitAlgorithm tree_split_mode,
                             int max_features,
                             int n_jobs = 1,
                             int random_seed = -1);

  virtual ~RandomForestClassification();

  /**
   * @brief
   * \if Japanese
   * ランダムフォレストの学習
   * \else
   * Train the random forest model
   * \endif
   */
  virtual void fit(const std::vector<double> &X,
                   const std::vector<double> &Y,
                   std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * クラスラベルを予測する
   * \else
   * Predict class labels
   * \endif
   */
  virtual std::vector<double> predict(const std::vector<double> &X,
                                      std::size_t l, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 各クラスの確率を予測する
   * \else
   * Predict class probabilities
   * \endif
   */
  virtual std::vector<double> predict_proba(const std::vector<double> &X,
                                            std::size_t l, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 各特徴量の重要度を計算する
   * \else
   * Compute feature importance scores
   * \endif
   */
  std::vector<double> feature_importances() const;

  // --- 各ハイパーパラメータのゲッター ---
  int get_n_trees() const;
  int get_max_depth() const;
  int get_min_samples_leaf() const;
  int get_min_samples_split() const;
  double get_min_impurity_decrease() const;
  int get_max_bins() const;
  int get_max_features() const;
  Criterion get_tree_mode() const;
  SplitAlgorithm get_tree_split_mode() const;
  int get_n_jobs() const;
  const int check_initialize() { return is_initialized_; }

private:
  int n_trees_;
  int max_depth_;
  int min_samples_leaf_;
  int min_samples_split_;
  double min_impurity_decrease_;
  int max_leaf_nodes_;
  int max_bins_;
  int max_features_;
  Criterion tree_mode_;
  SplitAlgorithm tree_split_mode_;
  int n_jobs_;
  int num_features_;
  bool is_initialized_;
  int random_seed_;

  /**
   * @brief
   * \if Japanese
   * ランダムフォレストに含まれる決定木（スマートポインタで管理）
   * \else
   * Decision trees contained in the random forest (managed with smart pointers)
   * \endif
   */
  std::vector<std::unique_ptr<DecisionTreeClassification>> trees_;

  /**
   * @brief
   * \if Japanese
   * 乱数エンジン（ブートストラップサンプル用）
   * \else
   * Random number engine (for bootstrap sampling)
   * \endif
   */
  mutable std::mt19937 rng_;

  /**
   * @brief
   * \if Japanese
   * 各決定木を学習させる（並列処理対応）
   * \else
   * Train each decision tree (parallel processing supported)
   * \endif
   */
  void fit_tree(const std::vector<double> &X,
                const std::vector<double> &Y,
                std::size_t n, std::size_t m,
                int tree_index,
                const std::vector<int> &bootstrap_indices);
};

#endif // RANDOM_FOREST_CLASSIFICATION_HPP
