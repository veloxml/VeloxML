#ifndef KMEANS_HPP
#define KMEANS_HPP

#include "base/unsupervised_base.hpp"
#include <vector>
#include <cstddef>

/**
 * @enum KMeansAlgorithm
 * @brief
 * \if Japanese
 * K-Means アルゴリズムの種別を表す列挙型
 *
 * Lloyd の標準アルゴリズム、Elkan の高速アルゴリズム、Hamerly の高速アルゴリズムを選択可能。
 * \else
 * Enumeration representing the types of K-Means algorithms
 *
 * Supports standard Lloyd's algorithm, Elkan's optimized algorithm, and Hamerly's optimized algorithm.
 * \endif
 */
enum class KMeansAlgorithm
{
  STANDARD, ///< \if Japanese 標準の Lloyd's algorithm \else Standard Lloyd's algorithm \endif
  ELKAN,    ///< \if Japanese Elkan の最適化アルゴリズム \else Elkan’s optimized algorithm \endif
  HAMERLY   ///< \if Japanese Hamerly の最適化アルゴリズム \else Hamerly’s optimized algorithm \endif
};

/**
 * @class KMeans
 * @brief
 * \if Japanese
 * K-Means クラスタリングの実装
 *
 * K-Means は、与えられたデータを `n_clusters` 個のクラスタに分割する教師なし学習アルゴリズム。
 * Lloyd の標準アルゴリズム、Elkan の高速アルゴリズム、Hamerly の高速アルゴリズムを選択できる。
 * \else
 * Implementation of K-Means clustering
 *
 * K-Means is an unsupervised learning algorithm that partitions the given data into `n_clusters` clusters.
 * Supports standard Lloyd's algorithm, Elkan's optimized algorithm, and Hamerly's optimized algorithm.
 * \endif
 */
class KMeans : public UnsupervisedEstimatorBase
{
public:
  /**
   * @brief
   * \if Japanese
   * K-Means クラスタリングのコンストラクタ
   *
   * クラスタ数、最大イテレーション数、収束判定閾値、使用するアルゴリズム、KD-Tree の有無を設定する。
   * \else
   * Constructor for K-Means clustering
   *
   * Configures the number of clusters, maximum iterations, convergence tolerance, algorithm type, and whether to use KD-Tree.
   * \endif
   *
   * @param n_clusters
   * \if Japanese クラスタ数 \else Number of clusters \endif
   *
   * @param max_iter
   * \if Japanese 最大イテレーション数（デフォルト: 300） \else Maximum number of iterations (default: 300) \endif
   *
   * @param tol
   * \if Japanese 収束判定閾値（デフォルト: 1e-4） \else Convergence tolerance (default: 1e-4) \endif
   *
   * @param algorithm
   * \if Japanese 使用する K-Means アルゴリズム（デフォルト: STANDARD） \else K-Means algorithm type to use (default: STANDARD) \endif
   *
   * @param use_kdtree
   * \if Japanese KD-Tree を使用するか（デフォルト: false） \else Whether to use KD-Tree (default: false) \endif
   */
  KMeans(int n_clusters, int max_iter = 300, double tol = 1e-4,
         KMeansAlgorithm algorithm = KMeansAlgorithm::STANDARD,
         bool use_kdtree = false);

  /**
   * @brief
   * \if Japanese
   * デストラクタ
   * \else
   * Destructor
   * \endif
   */
  ~KMeans();

  /**
   * @brief
   * \if Japanese
   * モデルを学習する
   *
   * データ `X` を `n_clusters` 個のクラスタに分類する。
   * \else
   * Train the model
   *
   * Clusters the data `X` into `n_clusters` clusters.
   * \endif
   */
  void fit(const std::vector<double> &X, std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * クラスタラベルを予測する
   *
   * 入力データ `X` の各点について、最も近いクラスタのラベルを返す。
   * \else
   * Predict cluster labels
   *
   * Returns the label of the nearest cluster for each point in the input data `X`.
   * \endif
   */
  std::vector<double> predict(const std::vector<double> &X, std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 各データ点とクラスタ中心間の距離を計算する
   * \else
   * Compute distances between each data point and cluster centroids
   * \endif
   */
  std::vector<double> transform(const std::vector<double> &X, std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * モデルを学習した後に transform を実行する
   * \else
   * Train the model and then execute transform
   * \endif
   */
  std::vector<double> fit_transform(const std::vector<double> &X, std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * クラスタ中心を取得する
   * \else
   * Get cluster centroids
   * \endif
   */
  const std::vector<double> &get_centroids() const { return centroids_; }

  const bool check_initialize() const { return is_initialized_; }

private:
  int n_clusters_;            ///< \if Japanese クラスタ数 \else Number of clusters \endif
  int max_iter_;              ///< \if Japanese 最大イテレーション数 \else Maximum number of iterations \endif
  double tol_;                ///< \if Japanese 収束判定閾値 \else Convergence tolerance \endif
  KMeansAlgorithm algorithm_; ///< \if Japanese 使用する K-Means アルゴリズム \else K-Means algorithm type used \endif
  bool use_kdtree_;           ///< \if Japanese KD-Tree を使用するか \else Whether KD-Tree is used \endif
  bool is_initialized_;          ///< \if Japanese 学習済みフラグ \else Flag of Fitted \endif

  std::vector<double> centroids_; ///< \if Japanese クラスタ中心（k × m の Col-Major 行列） \else Cluster centroids (k × m Col-Major matrix) \endif

  /**
   * @brief
   * \if Japanese
   * クラスタ中心を初期化する
   * \else
   * Initialize cluster centroids
   * \endif
   */
  void initialize_centroids(const std::vector<double> &X, std::size_t n, std::size_t m);

  /**
   * @brief
   * \if Japanese
   * 標準的な K-Means (Lloyd's Algorithm) の実行
   * \else
   * Execute standard K-Means (Lloyd's Algorithm)
   * \endif
   */
  void run_standard(const std::vector<double> &X, std::size_t n, std::size_t m);

  /**
   * @brief
   * \if Japanese
   * Elkan の最適化アルゴリズムの実行
   * \else
   * Execute Elkan's optimized algorithm
   * \endif
   */
  void run_elkan(const std::vector<double> &X, std::size_t n, std::size_t m);

  /**
   * @brief
   * \if Japanese
   * Hamerly の最適化アルゴリズムの実行
   * \else
   * Execute Hamerly's optimized algorithm
   * \endif
   */
  void run_hamerly(const std::vector<double> &X, std::size_t n, std::size_t m);

  /**
   * @brief
   * \if Japanese
   * 2つのデータ点間のユークリッド距離の二乗を計算する
   *
   * 与えられた2つの `m` 次元ベクトル `a` と `b` のユークリッド距離の二乗を計算する。
   * \else
   * Compute the squared Euclidean distance between two data points
   *
   * Computes the squared Euclidean distance between two `m`-dimensional vectors `a` and `b`.
   * \endif
   *
   * @param a
   * \if Japanese
   * ベクトル a（m次元）
   * \else
   * Vector a (m-dimensional)
   * \endif
   *
   * @param b
   * \if Japanese
   * ベクトル b（m次元）
   * \else
   * Vector b (m-dimensional)
   * \endif
   *
   * @param m
   * \if Japanese
   * 次元数
   * \else
   * Number of dimensions
   * \endif
   *
   * @return
   * \if Japanese
   * ユークリッド距離の二乗
   * \else
   * Squared Euclidean distance
   * \endif
   */
  double sq_euclidean_distance(const double *a, const double *b, std::size_t m);
};

#endif // KMEANS_HPP
