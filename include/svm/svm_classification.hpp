#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H

#include "base/classification_base.hpp"
#include <vector>
#include <string>
#include <cstddef>
#include <random>

// Platt scaling 用の LogisticRegressionClass
#include "linear/logistic_regression.hpp"

/**
 * @class SVMClassification
 * @brief
 * \if Japanese
 * サポートベクターマシン (Support Vector Machine, SVM) の分類器
 *
 * ソフトマージン SVM に対応し、カーネル手法（線形、RBF、多項式）を選択可能。
 * 近似カーネル法（例：ランダムフーリエ特徴）を使用することもできる。
 * \else
 * Support Vector Machine (SVM) Classifier
 *
 * Supports soft-margin SVM with selectable kernel methods (linear, RBF, polynomial).
 * Approximate kernel methods (e.g., Random Fourier Features) can also be used.
 * \endif
 */
class SVMClassification : public ClassificationBase
{
public:
  /**
   * @brief
   * \if Japanese
   * SVM分類器のコンストラクタ
   * \else
   * Constructor for SVM classifier
   * \endif
   *
   * @param C
   * \if Japanese ソフトマージンのパラメータ（ハードマージンの場合は十分大きな値を設定） \else Soft margin parameter (set a sufficiently large value for hard margin) \endif
   *
   * @param tol
   * \if Japanese 収束判定用の許容誤差 \else Convergence tolerance \endif
   *
   * @param max_passes
   * \if Japanese 収束までの最大パス数 \else Maximum number of passes until convergence \endif
   *
   * @param kernel
   * \if Japanese カーネルの種類 ("linear", "rbf", "poly") \else Kernel type ("linear", "rbf", "poly") \endif
   *
   * @param gamma_scale
   * \if Japanese `true` の場合、gamma を自動スケーリング \else If `true`, gamma is automatically scaled \endif
   *
   * @param gamma
   * \if Japanese RBF カーネル用パラメータ（poly カーネルでも内積計算に使用される場合あり） \else Parameter for RBF kernel (may also be used in polynomial kernel) \endif
   *
   * @param coef0
   * \if Japanese 多項式カーネルの定数項 \else Constant term for polynomial kernel \endif
   *
   * @param degree
   * \if Japanese 多項式カーネルの次数 \else Degree of the polynomial kernel \endif
   *
   * @param approx_kernel
   * \if Japanese 近似カーネル法を利用する場合 `true` \else Set `true` to use an approximate kernel method \endif
   */
  SVMClassification(double C,
                    double tol,
                    int max_passes,
                    const std::string &kernel,
                    bool gamma_scale = true,
                    double gamma = 0.1,
                    double coef0 = 0.0,
                    int degree = 3,
                    bool approx_kernel = false);

  virtual ~SVMClassification();

  /**
   * @brief
   * \if Japanese
   * モデルの学習
   * \else
   * Train the model
   * \endif
   */
  virtual void fit(const std::vector<double> &X,
                   const std::vector<double> &Y,
                   std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 予測（ラベルを返す）
   * \else
   * Predict labels
   * \endif
   */
  virtual std::vector<double> predict(const std::vector<double> &X,
                                      std::size_t l, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 予測確率を推定する（Platt Scaling 使用）
   * \else
   * Estimate probabilities (using Platt Scaling)
   * \endif
   */
  virtual std::vector<double> predict_proba(const std::vector<double> &X,
                                            std::size_t l, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 決定関数のスコアを返す
   * \else
   * Return decision function scores
   * \endif
   */
  std::vector<double> predict_score(const std::vector<double> &X,
                                    std::size_t l, std::size_t m);

  // --- ハイパーパラメータのゲッター ---
  double getC() const;
  double getTol() const;
  int getMaxPasses() const;
  std::string getKernel() const;
  double getGamma() const;
  double getCoef0() const;
  int getDegree() const;
  bool getApproxKernel() const;

  const bool check_initialize() { return is_initialized_; }

private:
  // ハイパーパラメータ
  double C_;
  double tol_;
  int max_passes_;
  std::string kernel_;
  bool gamma_scale_;
  double gamma_;
  double coef0_;
  int degree_;
  bool approx_kernel_;
  bool is_initialized_;

  // 学習済みパラメータ
  std::vector<double> alphas_;
  double b_;
  std::vector<double> w_;

  // エラーキャッシュ
  std::vector<double> errors_;

  std::vector<double> sample_norms_;
  std::vector<double> kernel_cache_;
  std::vector<double> active_set_;

  // 学習データ（ColMajor形式）
  std::vector<double> X_train_;
  std::vector<double> Y_train_;
  std::size_t n_train_;
  std::size_t m_features_;

  // 近似カーネル法用パラメータ（例：ランダムフーリエ特徴）
  std::vector<double> rff_weights_;
  std::vector<double> rff_bias_;
  std::size_t rff_dim_;

  // Platt scaling 用のキャリブレーションモデル
  LogisticRegression platt_model_;
  bool is_platt_calibrated_;

  /**
   * @brief
   * \if Japanese
   * インデックス i, j に対してカーネル値を計算する
   * \else
   * Compute kernel value for sample indices i, j
   * \endif
   */
  double kernel_function_index(std::size_t i, std::size_t j) const;

  /**
   * @brief
   * \if Japanese
   * 2つのベクトル間のカーネル計算（連続メモリ上のデータを前提）
   * \else
   * Compute kernel value between two vectors (assumes contiguous memory)
   * \endif
   */
  double kernel_function(const double *x1, const double *x2) const;

  /**
   * @brief
   * \if Japanese
   * 近似カーネル法により入力データを変換する（Random Fourier Features など）
   * \else
   * Transform input data using approximate kernel methods (e.g., Random Fourier Features)
   * \endif
   */
  void compute_approx_features(const std::vector<double> &X,
                               std::vector<double> &X_approx,
                               std::size_t l, std::size_t m) const;

  /**
   * @brief
   * \if Japanese
   * BLAS/OpenBLAS を用いたドット積計算
   * \else
   * Compute dot product using BLAS/OpenBLAS
   * \endif
   */
  double dot(const double *a, const double *b, std::size_t len) const;
};

#endif // SVM_CLASSIFIER_H
