#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include "base/classification_base.hpp"
#include "solver/lbfgs.hpp"
#include "solver/newton_cg.hpp"
#include "linear/logistic_regression.hpp"
#include <vector>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

/**
 * @enum LogisticRegressionSolverType
 * @brief
 * \if Japanese
 * ロジスティック回帰のソルバー種別
 *
 * L-BFGS, Newton法, 座標降下法 (Coordinate Descent) のいずれかを選択可能。
 * \else
 * Types of solvers for logistic regression
 *
 * Supports L-BFGS, Newton's method, and Coordinate Descent.
 * \endif
 */
enum class LogisticRegressionSolverType
{
  LBFGS,  ///< \if Japanese L-BFGS \else L-BFGS \endif
  NEWTON, ///< \if Japanese Newton法 \else Newton's method \endif
  CD      ///< \if Japanese 座標降下法 (Coordinate Descent) \else Coordinate Descent \endif
};

#include <stdexcept>

/**
 * @class LogisticRegressionL2Objective
 * @brief
 * \if Japanese
 * L2 正則化付きロジスティック回帰の目的関数
 *
 * ロジスティック回帰の損失関数を L2 正則化付きで計算し、勾配・ヘッセ行列を求める。
 * \else
 * Logistic Regression Objective with L2 Regularization
 *
 * Computes the logistic regression loss function with L2 regularization, including gradient and Hessian calculations.
 * \endif
 */
class LogisticRegressionL2Objective
{
public:
  /**
   * @brief
   * \if Japanese
   * コンストラクタ
   * \else
   * Constructor
   * \endif
   */
  LogisticRegressionL2Objective(const std::vector<double> &X_config, int n, int m,
                                const std::vector<double> &y,
                                double lambda);

  /**
   * @brief
   * \if Japanese
   * 損失関数とその勾配の計算
   * \else
   * Compute loss function and its gradient
   * \endif
   */
  double operator()(const std::vector<double> &theta, std::vector<double> &grad);

  /**
   * @brief
   * \if Japanese
   * 勾配の計算
   * \else
   * Compute the gradient
   * \endif
   */
  std::vector<double> gradient(const std::vector<double> &theta);

  /**
   * @brief
   * \if Japanese
   * 完全なヘッセ行列の計算
   * \else
   * Compute the full Hessian matrix
   * \endif
   */
  std::vector<std::vector<double>> hessian(const std::vector<double> &theta);

  /**
   * @brief Hessian-vector 積を計算する
   *        H*v = X^T [ p(1-p) .* (X*v) ] + lambda*v,
   *        ただし p = 1/(1+exp(-X*theta))
   * @param theta 現在のパラメータ（サイズ m）
   * @param v 入力ベクトル（サイズ m）
   * @param hv 出力：Hessian-vector 積の結果（サイズ m）
   */
  void hvp(const std::vector<double> &theta,
           const std::vector<double> &v,
           std::vector<double> &hv);

private:
  const std::vector<double> &X_;
  const std::vector<double> &y_;
  double lambda_;
  size_t N_, D_;
};

/**
 * @class LogisticRegression
 * @brief
 * \if Japanese
 * ロジスティック回帰モデルの実装
 *
 * ロジスティック回帰を L-BFGS, Newton 法, 座標降下法のいずれかで解く。
 * \else
 * Implementation of Logistic Regression Model
 *
 * Solves logistic regression using L-BFGS, Newton's method, or Coordinate Descent.
 * \endif
 */
class LogisticRegression : public ClassificationBase
{
public:
  /**
   * @brief
   * \if Japanese
   * ロジスティック回帰モデルのコンストラクタ
   * \else
   * Constructor for Logistic Regression Model
   * \endif
   * @param solver       ソルバー種別
   * @param lambda       正則化パラメータ
   * @param tol          収束許容誤差
   * @param maxIter      最大反復回数
   * @param ls_alpha_init 初期ステップ長
   * @param ls_rho       ラインサーチの減衰率
   * @param ls_c         Armijo条件定数
   * @param history_size LBFGSの履歴サイズ
   * @param lbfgs_k      LBFGS用のパラメータ
   */
  LogisticRegression(
      LogisticRegressionSolverType solver = LogisticRegressionSolverType::LBFGS,
      double lambda = 1.0,
      double tol = 1e-6,
      int maxIter = 100,
      double ls_alpha_init = 1.0,
      double ls_rho = 0.5,
      double ls_c = 1e-4,
      int history_size = 10,
      int lbfgs_k = 1);

  /**
   * @brief
   * \if Japanese
   * モデルを学習する
   * \else
   * Train the model
   * \endif
   */
  void fit(const std::vector<double> &X,
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
  std::vector<double> predict(const std::vector<double> &X,
                              std::size_t l, std::size_t m) override;

  /**
   * @brief
   * \if Japanese
   * クラス確率を予測する
   * \else
   * Predict class probabilities
   * \endif
   */
  std::vector<double> predict_proba(const std::vector<double> &X,
                                    std::size_t l, std::size_t m) override;

  // // pybind11用オーバーロード（入出力はpy::array_t<double>）
  // pybind11::array_t<double> fit(const pybind11::array_t<double> &X,
  //                               const pybind11::array_t<double> &Y);
  // pybind11::array_t<double> predict(const pybind11::array_t<double> &X);
  // pybind11::array_t<double> predict_proba(const pybind11::array_t<double> &X);

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
   * 学習済みのバイアスを取得する
   * \else
   * Get the learned bias
   * \endif
   */
  double get_bias() const;

  /**
   * @brief
   * \if Japanese
   * 使用しているソルバーの種類を取得する
   * \else
   * Get the type of solver being used
   * \endif
   */
  LogisticRegressionSolverType get_solver() const;

  /**
   * @brief
   * \if Japanese
   * 正則化パラメータを取得する
   * \else
   * Get the regularization parameter
   * \endif
   */
  double get_lambda() const;

  /**
   * @brief
   * \if Japanese
   * 収束許容誤差を取得する
   * \else
   * Get the convergence tolerance
   * \endif
   */
  double get_tol() const;

  /**
   * @brief
   * \if Japanese
   * 最大反復回数を取得する
   * \else
   * Get the maximum number of iterations
   * \endif
   */
  int get_maxIter() const;

  double get_ls_alpha_init() const;
  double get_ls_rho() const;
  double get_ls_c() const;
  int get_history_size() const;
  int get_lbfgs_k() const;

private:
  // 各ソルバーの内部実装
  void fit_lbfgs(const std::vector<double> &X,
                 const std::vector<double> &Y,
                 std::size_t n, std::size_t m);
  void fit_newton(const std::vector<double> &X,
                  const std::vector<double> &Y,
                  std::size_t n, std::size_t m);
  void fit_cd(const std::vector<double> &X,
              const std::vector<double> &Y,
              std::size_t n, std::size_t m);

  // 学習パラメータ（weights, bias）およびハイパーパラメータ
  std::vector<double> weights_;
  double bias_;
  LogisticRegressionSolverType solver_;
  double lambda_;
  double tol_;
  int maxIter_;
  double ls_alpha_init_;
  double ls_rho_;
  double ls_c_;
  int history_size_;
  int lbfgs_k_;
};

#endif // LOGISTIC_REGRESSION_HPP
