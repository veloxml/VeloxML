#ifndef DECISION_TREE_CLASSIFICATION_HPP
#define DECISION_TREE_CLASSIFICATION_HPP

#include "base/classification_base.hpp" // 先に示した ClassificationBase のヘッダー
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * @enum Criterion
 * @brief
 * \if Japanese
 * 決定木の不純度評価基準
 * \else
 * Impurity criteria for decision trees
 * \endif
 */
enum class Criterion
{
  Entropy, ///< \if Japanese エントロピー \else Entropy \endif
  Gini,    ///< \if Japanese ジニ不純度 \else Gini impurity \endif
  Logloss  ///< \if Japanese ログロス（対数損失） \else Log-loss (logarithmic loss) \endif
};

/**
 * @enum SplitAlgorithm
 * @brief
 * \if Japanese
 * 分割アルゴリズムの種類
 * \else
 * Types of splitting algorithms
 * \endif
 */
enum class SplitAlgorithm
{
  Standard, ///< \if Japanese 標準的な分割 \else Standard split \endif
  Histogram ///< \if Japanese ヒストグラムベースの分割 \else Histogram-based split \endif
};

/**
 * @class DecisionTreeClassification
 * @brief
 * \if Japanese
 * 決定木分類器
 *
 * 指定された分割基準（Gini, Entropy, Logloss）と分割アルゴリズム（Standard, Histogram）を用いて
 * 決定木を学習し、分類タスクを行う。
 * \else
 * Decision Tree Classifier
 *
 * A classification tree that uses the specified impurity criterion (Gini, Entropy, Logloss)
 * and split algorithm (Standard, Histogram) for learning.
 * \endif
 */
class DecisionTreeClassification : public ClassificationBase
{
public:
  /**
   * @brief
   * \if Japanese
   * 決定木分類器のコンストラクタ
   * \else
   * Constructor for Decision Tree Classifier
   * \endif
   *
   * @param max_depth
   * \if Japanese 最大の木の深さ \else Maximum tree depth \endif
   *
   * @param min_samples_split
   * \if Japanese ノード分割に必要な最小サンプル数 \else Minimum number of samples required to split a node \endif
   *
   * @param max_bins
   * \if Japanese ヒストグラムベース分割のための最大ビン数 \else Maximum number of bins for histogram-based splits \endif
   *
   * @param criterion
   * \if Japanese 不純度の評価基準（デフォルトは Gini） \else Impurity criterion (default: Gini) \endif
   *
   * @param split_algorithm
   * \if Japanese 分割アルゴリズム（デフォルトは Standard） \else Splitting algorithm (default: Standard) \endif
   *
   * @param min_samples_leaf
   * \if Japanese 葉ノードに含まれる最小サンプル数 \else Minimum number of samples in a leaf node \endif
   *
   * @param max_leaf_nodes
   * \if Japanese 最大の葉ノード数 \else Maximum number of leaf nodes \endif
   *
   * @param min_impurity_decrease
   * \if Japanese ノード分割時に必要な最小不純度減少量 \else Minimum impurity decrease required for a node split \endif
   *
   * @param max_features
   * \if Japanese 各分割で考慮する最大特徴量数 \else Maximum number of features considered per split \endif
   */
  DecisionTreeClassification(int max_depth, int min_samples_split, int max_bins,
                             Criterion criterion = Criterion::Gini,
                             SplitAlgorithm split_algorithm = SplitAlgorithm::Standard,
                             int min_samples_leaf = 1, int max_leaf_nodes = 5,
                             double min_impurity_decrease = 0.0, int max_features = 5);
  virtual ~DecisionTreeClassification();

  /**
   * @brief
   * \if Japanese
   * 決定木を学習する
   * \else
   * Train the decision tree
   * \endif
   */
  virtual void fit(const std::vector<double> &X,
                   const std::vector<double> &Y,
                   std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 入力データの分類を行う
   * \else
   * Perform classification on input data
   * \endif
   */
  virtual std::vector<double> predict(const std::vector<double> &X,
                                      std::size_t l, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 各クラスの確率を予測する
   * \else
   * Predict probabilities for each class
   * \endif
   */
  virtual std::vector<double> predict_proba(const std::vector<double> &X,
                                            std::size_t l, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 特徴量の重要度を計算する
   * \else
   * Compute feature importance
   * \endif
   */
  std::vector<double> compute_feature_importance() const;

private:
  int max_depth_;
  int min_samples_split_;
  int max_bins_;
  Criterion criterion_;
  SplitAlgorithm split_algorithm_;
  int min_samples_leaf_;
  int max_leaf_nodes_;
  double min_impurity_decrease_;
  int current_leaf_count_;
  int max_features_;
  int num_features_;

  /**
   * @struct Node
   * @brief
   * \if Japanese
   * 決定木のノード構造
   * \else
   * Structure representing a node in the decision tree
   * \endif
   */
  struct Node
  {
    int feature_index;                 ///< \if Japanese 分割特徴量のインデックス \else Index of the feature used for splitting \endif
    double threshold;                  ///< \if Japanese 分割の閾値 \else Threshold for splitting \endif
    int left;                          ///< \if Japanese 左の子ノードのインデックス \else Index of the left child node \endif
    int right;                         ///< \if Japanese 右の子ノードのインデックス \else Index of the right child node \endif
    bool is_leaf;                      ///< \if Japanese 葉ノードかどうか \else Whether the node is a leaf \endif
    int predicted_class;               ///< \if Japanese 予測クラス（葉ノードの場合） \else Predicted class (for leaf nodes) \endif
    std::vector<double> probabilities; ///< \if Japanese 各クラスの確率 \else Class probabilities \endif
    int sample_count;                  ///< \if Japanese ノードに到達したサンプル数 \else Number of samples reaching this node \endif
    double impurity;                   ///< \if Japanese ノードの不純度 \else Impurity of the node \endif
  };

  std::vector<Node> tree_;

  // ヘッダファイル (decision_tree_classification.hpp) の private 領域に追加
  void accumulate_importance(int node_index, std::vector<double> &importances) const;

  // ツリー構築のための再帰的ヘルパー
  void find_best_threshold(
      const std::vector<double> &X, std::size_t n, std::size_t m,
      const std::vector<double> &Y, const std::vector<int> &indices,
      int feature_index, int num_classes,
      double &best_threshold, double &best_impurity,
      std::vector<int> &best_left_indices, std::vector<int> &best_right_indices) const;

  int build_tree(const std::vector<double> &X, std::size_t n, std::size_t m,
                 const std::vector<double> &Y, const std::vector<int> &indices, int depth);
  // 標準的な分割
  void split_data(const std::vector<double> &X, std::size_t n, std::size_t m,
                  const std::vector<int> &indices, int feature_index, double threshold,
                  std::vector<int> &left_indices, std::vector<int> &right_indices) const;
  // ヒストグラム近似分割（SIMD/並列処理付きの実装例）
  void split_data_histogram(const std::vector<double> &X, std::size_t n, std::size_t m,
                            const std::vector<int> &indices, int feature_index, double threshold,
                            std::vector<int> &left_indices, std::vector<int> &right_indices) const;
  // 不純度計算
  double compute_impurity(const std::vector<int> &indices, const std::vector<double> &Y,
                          int num_classes) const;
  double compute_impurity_entropy(const std::vector<int> &indices, const std::vector<double> &Y,
                                  int num_classes) const;
  double compute_impurity_gini(const std::vector<int> &indices, const std::vector<double> &Y,
                               int num_classes) const;
  double compute_impurity_logloss(const std::vector<int> &indices, const std::vector<double> &Y,
                                  int num_classes) const;
  // Y に含まれるクラス数（最大ラベル値＋1 として計算）
  int get_num_classes(const std::vector<double> &Y) const;
};

#endif // DECISION_TREE_CLASSIFICATION_HPP
