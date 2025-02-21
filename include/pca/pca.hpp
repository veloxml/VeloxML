#ifndef PCA_HPP
#define PCA_HPP

#include "base/unsupervised_base.hpp"
#include <vector>

/**
 * @class PCA
 * @brief
 * \if Japanese
 * 主成分分析（Principal Component Analysis, PCA）の実装クラス
 *
 * 入力データの次元削減を行う手法であり、データの分散が最大となる軸（主成分）を見つける。
 * \else
 * Implementation of Principal Component Analysis (PCA)
 *
 * A dimensionality reduction technique that finds the principal components where variance is maximized.
 * \endif
 */
class PCA : public UnsupervisedEstimatorBase
{
public:
  /**
   * @brief
   * \if Japanese
   * PCAのコンストラクタ
   *
   * 出力する主成分の数を指定してPCAオブジェクトを生成する。
   * \else
   * Constructor for PCA
   *
   * Initializes a PCA object with the specified number of principal components.
   * \endif
   *
   * @param n_components
   * \if Japanese 取得する主成分の数 \else Number of principal components to retain \endif
   */
  PCA(int n_components);

  /**
   * @brief デストラクタ
   */
  ~PCA();

  /**
   * @brief
   * \if Japanese
   * PCAモデルの学習を行う
   *
   * 入力データ `X` を用いて、主成分を求める。
   * \else
   * Train the PCA model
   *
   * Finds the principal components using input data `X`.
   * \endif
   *
   * @param X
   * \if Japanese 入力データ（n_samples × n_features のCol-Major配列） \else Input data (n_samples × n_features Col-Major array) \endif
   *
   * @param n
   * \if Japanese サンプル数（行数） \else Number of samples (rows) \endif
   *
   * @param m
   * \if Japanese 特徴量の数（列数） \else Number of features (columns) \endif
   */
  void fit(const std::vector<double> &X, std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 入力データを主成分空間に変換する
   *
   * 学習済みのPCAモデルを用いてデータを次元削減する。
   * \else
   * Transform input data into the principal component space
   *
   * Uses the trained PCA model to reduce the dimensionality of input data.
   * \endif
   *
   * @param X
   * \if Japanese 入力データ（n_samples × n_features のCol-Major配列） \else Input data (n_samples × n_features Col-Major array) \endif
   *
   * @param n
   * \if Japanese サンプル数（行数） \else Number of samples (rows) \endif
   *
   * @param m
   * \if Japanese 特徴量の数（列数） \else Number of features (columns) \endif
   *
   * @return
   * \if Japanese 変換後のデータ（n_samples × n_components のCol-Major配列） \else Transformed data (n_samples × n_components Col-Major array) \endif
   */
  std::vector<double> transform(const std::vector<double> &X, std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 主成分分析の結果を用いたデータの変換（predictとtransformは同じ動作）
   * \else
   * Transform input data using the principal component analysis results (same as transform)
   * \endif
   */
  std::vector<double> predict(const std::vector<double> &X, std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * PCAモデルの学習と変換を同時に行う
   *
   * `fit` を実行した後、`transform` を適用する。
   * \else
   * Perform PCA training and transformation simultaneously
   *
   * Executes `fit` followed by `transform`.
   * \endif
   *
   * @return
   * \if Japanese 変換後のデータ（n_samples × n_components のCol-Major配列） \else Transformed data (n_samples × n_components Col-Major array) \endif
   */
  std::vector<double> fit_transform(const std::vector<double> &X, std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 取得する主成分の数を取得する
   * \else
   * Get the number of principal components
   * \endif
   */
  int get_n_components() const { return n_components_; }

  /**
   * @brief
   * \if Japanese
   * 平均ベクトルを取得する
   * \else
   * Get the mean vector
   * \endif
   */
  const std::vector<double> &get_mean() const { return mean_; }

  /**
   * @brief
   * \if Japanese
   * 主成分行列を取得する
   *
   * 各行が特徴量、各列が主成分となる行列を返す。
   * \else
   * Get the principal component matrix
   *
   * Returns a matrix where each row represents a feature and each column represents a principal component.
   * \endif
   */
  const std::vector<double> &get_components() const { return components_; }
  const bool check_initialize() const { return is_initialized_; }

private:
  int n_components_;               ///< \if Japanese 取得する主成分の数 \else Number of principal components \endif
  std::vector<double> mean_;       ///< \if Japanese 特徴量ごとの平均（サイズ: n_features） \else Mean of each feature (size: n_features) \endif
  std::vector<double> components_; ///< \if Japanese 主成分行列（サイズ: n_features × n_components, Col-Major） \else Principal component matrix (size: n_features × n_components, Col-Major) \endif
  bool is_initialized_;
};

#endif // PCA_HPP
