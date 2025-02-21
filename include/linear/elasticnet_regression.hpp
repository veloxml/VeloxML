#ifndef ELASTICNET_REGRESSION_HPP
#define ELASTICNET_REGRESSION_HPP

#include "base/regression_base.hpp" // 既存のベースクラス（実装済み）
#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

/**
 * @enum ElasticNetSolverMode
 * @brief
 * \if Japanese
 * Elastic Net回帰の解法を表す列挙型
 *
 * FISTA法と ADMM法のいずれかを選択可能。
 * \else
 * Enumeration representing solver methods for Elastic Net regression
 *
 * Supports FISTA method and ADMM method.
 * \endif
 */
enum class ElasticNetSolverMode
{
  FISTA, ///< \if Japanese FISTA 法を使用 \else Uses the FISTA method \endif
  ADMM   ///< \if Japanese ADMM 法を使用 \else Uses the ADMM method \endif
};

/**
 * @class ElasticNetRegression
 * @brief
 * \if Japanese
 * Elastic Net回帰の実装クラス
 *
 * Elastic Net回帰は、L1正則化（Lasso）とL2正則化（Ridge）を組み合わせた線形回帰モデルであり、
 * スパース性を保ちつつ、特徴選択のバランスを取ることができる。
 * 入力行列 `X` にバイアス項を付加し、最適化問題を FISTA または ADMM を用いて解く。
 * \else
 * Implementation class for Elastic Net Regression
 *
 * Elastic Net regression is a linear regression model that combines L1 regularization (Lasso) and L2 regularization (Ridge),
 * balancing feature selection while maintaining sparsity.
 * Adds a bias term to the input matrix `X` and solves the optimization problem using either FISTA or ADMM.
 * \endif
 */
class ElasticnetRegression : public RegressionBase
{
public:
  /**
   * @brief
   * \if Japanese
   * Elastic Net回帰モデルのコンストラクタ
   *
   * 正則化パラメータ、最大反復回数、収束判定誤差、ソルバーの種類を指定して Elastic Net回帰モデルを初期化する。
   * \else
   * Constructor for the Elastic Net Regression model
   *
   * Initializes the Elastic Net regression model with the specified regularization parameters, maximum iterations,
   * convergence tolerance, and solver type.
   * \endif
   *
   * @param lambda1
   * \if Japanese L1 正則化パラメータ \else L1 regularization parameter \endif
   *
   * @param lambda2
   * \if Japanese L2 正則化パラメータ \else L2 regularization parameter \endif
   *
   * @param max_iter
   * \if Japanese 最大反復回数 \else Maximum number of iterations \endif
   *
   * @param tol
   * \if Japanese 収束判定誤差 \else Convergence tolerance \endif
   *
   * @param mode
   * \if Japanese 使用するソルバー（FISTA / ADMM、デフォルト: FISTA） \else Solver type (FISTA / ADMM, default: FISTA) \endif
   *
   * @param admm_rho
   * \if Japanese ADMM用パラメータ（FISTA では使用しない） \else Parameter for ADMM method (not used in FISTA) \endif
   *
   * @param penalize_bias
   * \if Japanese バイアス項を正則化するか（デフォルト: false） \else Whether to regularize the bias term (default: false) \endif
   */
  ElasticnetRegression(double lambda1, double lambda2, int max_iter, double tol,
                       ElasticNetSolverMode mode = ElasticNetSolverMode::FISTA,
                       double admm_rho = 1.0,
                       bool penalize_bias = false);

  /**
   * @brief
   * \if Japanese
   * モデルを学習する
   *
   * 説明変数 `X` (n × m) と目的変数 `Y` (n × 1) を用いて Elastic Net 回帰モデルを学習する。
   * \else
   * Train the model
   *
   * Trains the Elastic Net regression model using feature variables `X` (n × m) and target variables `Y` (n × 1).
   * \endif
   *
   * @param X
   * \if Japanese 入力データ（n × m のCol-Major配列） \else Input data (n × m Col-Major array) \endif
   *
   * @param Y
   * \if Japanese 目的変数（n × 1 の配列） \else Target variable (n × 1 array) \endif
   *
   * @param n
   * \if Japanese サンプル数（行数） \else Number of samples (rows) \endif
   *
   * @param m
   * \if Japanese 特徴量の数（列数） \else Number of features (columns) \endif
   */
  void fit(const std::vector<double> &X, const std::vector<double> &Y, std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * モデルによる予測を行う
   *
   * 入力データ `X` (n × m) に対して Elastic Net回帰の予測を行い、出力として (n × 1) のベクトルを返す。
   * \else
   * Perform prediction using the trained model
   *
   * Predicts outputs for the given input data `X` (n × m) and returns a (n × 1) vector.
   * \endif
   */
  std::vector<double> predict(const std::vector<double> &X, std::size_t n, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * 学習済みの重みを取得する
   * \else
   * Get the learned weights
   * \endif
   */
  std::vector<double> get_weights() const;

  /**
   * @brief
   * \if Japanese
   * 学習済みのバイアス項を取得する
   * \else
   * Get the learned bias term
   * \endif
   */
  double get_bias() const;

private:
  double lambda1_;
  double lambda2_;
  int max_iter_;
  double tol_;
  ElasticNetSolverMode solver_mode_;
  double admm_rho_;
  bool penalize_bias_;

  // 内部パラメータ：θ = [weights; bias]（サイズは m+1）
  std::vector<double> theta_;
  std::vector<double> weights_;
  double bias_;

  /**
   * @brief 入力行列 X にバイアス項（1.0の列）を付加する
   * @param X もとの (n×m) Col‐Major 配列
   * @return 付加後の (n×(m+1)) Col‐Major 配列
   */
  std::vector<double> augmentWithBias(const double *X, int n, int m);

  /**
   * @brief X_aug の転置×X_aug に λ₂ を加えた行列の最大固有値を計算し、Lipschitz 定数 L を求める
   * @param X_aug (n×d) の配列（d = m+1）
   * @param n サンプル数, d 次元
   * @return Lipschitz 定数
   */
  double computeLipschitzConstant(const std::vector<double> &X_aug, int n, int d);

  /**
   * @brief ソフト閾値演算（proximal operator）を、SIMD 最適化（OpenMP simd 指令等）付きで実施する
   * @param x 入力ベクトル（次元 d）
   * @param out 出力ベクトル（次元 d）
   * @param threshold 閾値
   * @param d 次元（通常 m+1）
   */
  void softThreshold(const std::vector<double> &x, std::vector<double> &out, double threshold, int d);

  /**
   * @brief FISTA ソルバーによる Elastic Net 解法
   * @param X もとの (n×m) Col‐Major 配列
   * @param Y (n×1) の配列
   */
  void solveFISTA(const double *X, int n, int m, const double *Y);

  /**
   * @brief ADMM ソルバーによる Elastic Net 解法
   * @param X もとの (n×m) Col‐Major 配列
   * @param Y (n×1) の配列
   */
  void solveADMM(const double *X, int n, int m, const double *Y);
};

#endif // ELASTICNET_REGRESSION_HPP
