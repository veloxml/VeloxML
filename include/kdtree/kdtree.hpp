#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <vector>
#include <memory>

/**
 * @struct KDTreeNode
 * @brief 
 * \if Japanese
 * k-d tree のノードを表す構造体
 * 
 * 各ノードは k-d tree 内の1点を表し、左右の部分木へのポインタを保持する。
 * \else
 * Structure representing a node in a k-d tree
 * 
 * Each node represents a point in the k-d tree and holds pointers to its left and right subtrees.
 * \endif
 */
struct KDTreeNode
{
  const double *point;               ///< \if Japanese ノードに格納する座標 \else Pointer to the coordinate stored in the node  \endif
  int index;                         ///< \if Japanese 元の点集合におけるインデックス \else Index in the original point set \endif
  int split_dim;                     ///< \if Japanese 分割軸（このノードで分割する次元） \else Split dimension (dimension used for partitioning at this node) \endif
  std::unique_ptr<KDTreeNode> left;  ///< \if Japanese 左部分木 \else Left subtree \endif
  std::unique_ptr<KDTreeNode> right; ///< \if Japanese 右部分木 \else Right subtree \endif
};

/**
 * @class KDTree
 * @brief 
 * \if Japanese
 * k-d tree の実装
 * 
 * k-d tree は高次元空間内の点を効率的に探索するためのデータ構造であり、
 * 最近傍探索などの用途に用いられる。
 * \else
 * Implementation of a k-d tree
 * 
 * A k-d tree is a data structure used for efficient searching in high-dimensional space, 
 * commonly used for nearest neighbor search.
 * \endif
 */
class KDTree
{
public:
  /**
   * @brief 
   * \if Japanese
   * k-d tree を構築するコンストラクタ
   * 
   * 与えられた点集合（Row-Major: `n × m`）から k-d tree を構築する。
   * \else
   * Constructor for building a k-d tree
   * 
   * Constructs a k-d tree from the given point set (Row-Major: `n × m`).
   * \endif
   * 
   * @param points 
   * \if Japanese
   * 入力点集合（Row-Major 配列）
   * \else
   * Input point set (Row-Major array)
   * \endif
   * 
   * @param n 
   * \if Japanese
   * 点の総数（行数）
   * \else
   * Total number of points (number of rows)
   * \endif
   * 
   * @param m 
   * \if Japanese
   * 次元数（列数）
   * \else
   * Number of dimensions (number of columns)
   * \endif
   */
  KDTree(const std::vector<double> &points, std::size_t n, std::size_t m);

  /**
   * @brief 
   * \if Japanese
   * 最近傍探索を行う
   * 
   * クエリ点 `query` に対して、最も近い点のインデックスを返す。
   * `bestDist` には最小の二乗距離が格納される。
   * \else
   * Perform nearest neighbor search
   * 
   * Finds the nearest point index for the given query point `query`.
   * The squared Euclidean distance of the nearest point is stored in `bestDist`.
   * \endif
   * 
   * @param query 
   * \if Japanese
   * クエリ点（m次元）
   * \else
   * Query point (m-dimensional)
   * \endif
   * 
   * @param bestDist 
   * \if Japanese
   * 最小の二乗距離（出力パラメータ）
   * \else
   * Squared Euclidean distance of the nearest point (output parameter)
   * \endif
   * 
   * @return 
   * \if Japanese
   * 最も近い点のインデックス
   * \else
   * Index of the nearest point
   * \endif
   */
  int nearestNeighbor(const double *query, double &bestDist) const;

  /**
   * @brief 
   * \if Japanese
   * SIMD を用いたユークリッド距離の二乗計算
   * \else
   * Compute squared Euclidean distance using SIMD
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
  static double sq_euclidean_distance(const double *a, const double *b, std::size_t m);

private:
  std::unique_ptr<KDTreeNode> root_; ///< \if Japanese k-d tree のルート \else Root of the k-d tree \endif
  std::size_t dimensions_;           ///< \if Japanese 点の次元数 \else Number of dimensions \endif

  /**
   * @brief 
   * \if Japanese
   * k-d tree の構築（再帰関数）
   * 
   * 点集合 `items` から深さ `depth` を基準にして k-d tree を構築する。
   * \else
   * Recursive function to build the k-d tree
   * 
   * Constructs a k-d tree from the given set of points `items`, based on the depth `depth`.
   * \endif
   * 
   * @param items 
   * \if Japanese
   * 点集合（インデックスと座標のペアのリスト）
   * \else
   * Set of points (list of index and coordinate pairs)
   * \endif
   * 
   * @param depth 
   * \if Japanese
   * 現在のツリーの深さ
   * \else
   * Current depth in the tree
   * \endif
   * 
   * @return 
   * \if Japanese
   * 構築された k-d tree のルートノード
   * \else
   * Root node of the constructed k-d tree
   * \endif
   */
  std::unique_ptr<KDTreeNode> buildTree(
      std::vector<std::pair<int, const double *>> items, int depth);

  /**
   * @brief 
   * \if Japanese
   * 最近傍探索（再帰関数）
   * 
   * `node` 以下の部分木について、クエリ点 `query` に最も近い点を探索する。
   * \else
   * Recursive function for nearest neighbor search
   * 
   * Searches for the nearest point to the query `query` within the subtree rooted at `node`.
   * \endif
   * 
   * @param node 
   * \if Japanese
   * 現在のノード
   * \else
   * Current node
   * \endif
   * 
   * @param query 
   * \if Japanese
   * クエリ点
   * \else
   * Query point
   * \endif
   * 
   * @param bestIndex 
   * \if Japanese
   * 現在の最適なインデックス（出力パラメータ）
   * \else
   * Current best index (output parameter)
   * \endif
   * 
   * @param bestDist 
   * \if Japanese
   * 最小の二乗距離（出力パラメータ）
   * \else
   * Minimum squared distance (output parameter)
   * \endif
   */
  void searchNearest(
      const KDTreeNode *node,
      const double *query,
      int &bestIndex,
      double &bestDist) const;
};

#endif // KDTREE_HPP
