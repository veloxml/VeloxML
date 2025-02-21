#ifndef LASSO_REGRESSION_HPP
#define LASSO_REGRESSION_HPP

#include "base/regression_base.hpp"
#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

/**
 * @enum LassoSolverMode
 * @brief
 * \if Japanese
 * Lasso回帰の解法を表す列挙型
 * 
 * FISTA法と ADMM法のいずれかを選択可能。
 * \else
 * Enumeration representing solver methods for Lasso regression
 * 
 * Supports FISTA method and ADMM method.
 * \endif
 */
enum class LassoSolverMode
{
    FISTA, ///< \if Japanese FISTA 法を使用 \else Uses the FISTA method \endif
    ADMM   ///< \if Japanese ADMM 法を使用 \else Uses the ADMM method \endif
};

/**
 * @class LassoRegression
 * @brief
 * \if Japanese
 * Lasso回帰の実装クラス
 * 
 * Lasso回帰は、L1正則化を加えた線形回帰モデルであり、スパースな解を求めることができる。
 * 入力行列 `X` にバイアス項を付加し、最適化問題を FISTA または ADMM を用いて解く。
 * \else
 * Implementation class for Lasso Regression
 * 
 * Lasso regression is a linear regression model with L1 regularization, enabling sparse solutions.
 * Adds a bias term to the input matrix `X` and solves the optimization problem using either FISTA or ADMM.
 * \endif
 */
class LassoRegression : public RegressionBase
{
public:
    /**
     * @brief
     * \if Japanese
     * Lasso回帰モデルのコンストラクタ
     * 
     * 正則化パラメータ、最大反復回数、収束判定誤差、ソルバーの種類を指定して Lasso回帰モデルを初期化する。
     * \else
     * Constructor for the Lasso Regression model
     * 
     * Initializes the Lasso regression model with the specified regularization parameter, maximum iterations,
     * convergence tolerance, and solver type.
     * \endif
     */
    LassoRegression(double lambda, int max_iter, double tol,
                    LassoSolverMode mode = LassoSolverMode::FISTA,
                    double admm_rho = 1.0,
                    bool penalize_bias = false);

    // --- C++ API ---

    /**
     * @brief
     * \if Japanese
     * モデルを学習する
     * 
     * 説明変数 `X` (n × m) と目的変数 `Y` (n × 1) を用いて Lasso 回帰モデルを学習する。
     * \else
     * Train the model
     * 
     * Trains the Lasso regression model using feature variables `X` (n × m) and target variables `Y` (n × 1).
     * \endif
     */
    void fit(const std::vector<double>& X, const std::vector<double>& Y, std::size_t n, std::size_t m) override;

    /**
     * @brief
     * \if Japanese
     * モデルによる予測を行う
     * 
     * 入力データ `X` (n × m) に対して Lasso回帰の予測を行い、出力として (n × 1) のベクトルを返す。
     * \else
     * Perform prediction using the trained model
     * 
     * Predicts outputs for the given input data `X` (n × m) and returns a (n × 1) vector.
     * \endif
     */
    std::vector<double> predict(const std::vector<double>& X, std::size_t n, std::size_t m) override;

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
    double lambda_;          ///< \if Japanese 正則化パラメータ \else Regularization parameter \endif
    int max_iter_;           ///< \if Japanese 最大反復回数 \else Maximum number of iterations \endif
    double tol_;             ///< \if Japanese 収束判定誤差 \else Convergence tolerance \endif
    LassoSolverMode solver_mode_; ///< \if Japanese 使用するソルバー（FISTA / ADMM） \else Solver type (FISTA / ADMM) \endif
    double admm_rho_;        ///< \if Japanese ADMM用パラメータ \else Parameter for ADMM method \endif
    bool penalize_bias_;     ///< \if Japanese バイアス項を正則化するか \else Whether to regularize the bias term \endif

    std::vector<double> theta_;   ///< \if Japanese 内部パラメータ（重みとバイアスを含む） \else Internal parameters (weights and bias) \endif
    std::vector<double> weights_; ///< \if Japanese 学習済みの重み（バイアス以外） \else Learned weights (excluding bias) \endif
    double bias_;            ///< \if Japanese 学習済みのバイアス項 \else Learned bias term \endif

    /**
     * @brief
     * \if Japanese
     * 入力行列 `X` にバイアス項を追加する
     * \else
     * Add a bias term to the input matrix `X`
     * \endif
     */
    std::vector<double> augmentWithBias(const double *X, int n, int m);

    /**
     * @brief
     * \if Japanese
     * FISTA による Lasso 解法
     * \else
     * Solve Lasso using FISTA method
     * \endif
     */
    void solveFISTA(const double *X, int n, int m, const double *Y);

    /**
     * @brief
     * \if Japanese
     * ADMM による Lasso 解法
     * \else
     * Solve Lasso using ADMM method
     * \endif
     */
    void solveADMM(const double *X, int n, int m, const double *Y);

    /**
     * @brief
     * \if Japanese
     * X_aug の転置×X_aug の最大固有値を求め、Lipschitz 定数 L を計算する
     * \else
     * Compute the maximum eigenvalue of X_aug^T * X_aug and determine the Lipschitz constant L
     * \endif
     */
    double computeLipschitzConstant(const std::vector<double> &X_aug, int n, int d);

    /**
     * @brief
     * \if Japanese
     * ソフト閾値演算（proximal operator）を、SIMD 最適化（OpenMP simd 指令を活用）付きで行う
     * \else
     * Perform soft-thresholding (proximal operator) with SIMD optimization (utilizing OpenMP SIMD directives)
     * \endif
     */
    void softThreshold(const std::vector<double> &x, std::vector<double> &out, double threshold, int d);
};

#endif // LASSO_REGRESSION_HPP
