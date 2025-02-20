#ifndef RIDGE_REGRESSION_HPP
#define RIDGE_REGRESSION_HPP

#include "base/regression_base.hpp"
#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// 並列化用 TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

/**
 * @class RidgeRegression
 * @brief
 * \if Japanese
 * Ridge回帰の実装クラス
 *
 * Ridge回帰は、最小二乗法に L2 正則化項を加えた回帰モデルであり、
 * 過学習を防ぐために使用される。
 * 正則化パラメータ `lambda` と、バイアス項の正則化を行うか (`penalize_bias`) を設定可能。
 * \else
 * Implementation class for Ridge Regression
 *
 * Ridge regression is a regression model with an L2 regularization term added to the least squares method,
 * used to prevent overfitting.
 * Supports configuration of the regularization parameter `lambda` and whether to regularize the bias term (`penalize_bias`).
 * \endif
 */
class RidgeRegression : public RegressionBase
{
public:
    /**
     * @brief
     * \if Japanese
     * Ridge回帰モデルのコンストラクタ
     *
     * 正則化パラメータとバイアス項の正則化の有無を指定して Ridge回帰モデルを初期化する。
     * \else
     * Constructor for the Ridge Regression model
     *
     * Initializes the Ridge regression model with the specified regularization parameter and whether to regularize the bias term.
     * \endif
     *
     * @param lambda
     * \if Japanese 正則化パラメータ（正の値） \else Regularization parameter (positive value) \endif
     *
     * @param penalize_bias
     * \if Japanese バイアス項を正則化するか（デフォルト: false） \else Whether to regularize the bias term (default: false) \endif
     */
    explicit RidgeRegression(double lambda, bool penalize_bias = false);

    // --- C++ API ---

    /**
     * @brief
     * \if Japanese
     * モデルを学習する
     *
     * 説明変数 `X` (n × m) と目的変数 `Y` (n × 1) を用いて Ridge 回帰モデルを学習する。
     * \else
     * Train the model
     *
     * Trains the Ridge regression model using feature variables `X` (n × m) and target variables `Y` (n × 1).
     * \endif
     */
    void fit(const std::vector<double> &X, const std::vector<double> &Y, std::size_t n, std::size_t m) override;

    /**
     * @brief
     * \if Japanese
     * モデルによる予測を行う
     *
     * 入力データ `X` (n × m) に対して Ridge回帰の予測を行い、出力として (n × 1) のベクトルを返す。
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
    double lambda_;               ///< \if Japanese 正則化パラメータ \else Regularization parameter \endif
    bool penalize_bias_;          ///< \if Japanese バイアス項を正則化するか \else Whether to regularize the bias term \endif
    std::vector<double> weights_; ///< \if Japanese 学習済みの重み（バイアス以外） \else Learned weights (excluding bias) \endif
    double bias_;                 ///< \if Japanese 学習済みのバイアス項 \else Learned bias term \endif

    /**
     * @brief
     * \if Japanese
     * 入力行列 `X` にバイアス項を追加する
     *
     * `X` (n × m) にバイアス項 (1.0 の列) を付加し、(n × (m+1)) のCol-Major配列として返す。
     * \else
     * Add a bias term to the input matrix `X`
     *
     * Appends a bias term (a column of 1.0) to `X` (n × m) and returns a Col-Major array of size (n × (m+1)).
     * \endif
     */
    std::vector<double> augmentWithBias(const double *X, int n, int m);

    /**
     * @brief
     * \if Japanese
     * Ridge回帰のパラメータを計算する
     *
     * Ridge 回帰の閉形式解を計算し、パラメータを推定する。
     * \else
     * Compute Ridge regression parameters
     *
     * Computes the closed-form solution of Ridge regression and estimates parameters.
     * \endif
     *
     * @param X
     * \if Japanese 元の行列 (n × m) のCol-Major配列 \else Original matrix (n × m) as a Col-Major array \endif
     *
     * @param Y
     * \if Japanese 目的変数 (n × 1) の配列 \else Target variable array (n × 1) \endif
     */
    void computeRegression(const double *X, int n, int m, const double *Y);

    /**
     * @brief
     * \if Japanese
     * 行列 `A` の対角成分に正則化項 `lambda` を加える
     *
     * `A` は (dim × dim) のCol-Major配列であり、対角成分に `lambda` を加える。
     * `penalize_bias` が `false` の場合、最後の対角成分（バイアス項）は変更しない。
     * \else
     * Add the regularization term `lambda` to the diagonal elements of matrix `A`
     *
     * `A` is a (dim × dim) Col-Major array, and `lambda` is added to the diagonal elements.
     * If `penalize_bias` is `false`, the last diagonal element (bias term) is not modified.
     * \endif
     *
     * @param A
     * \if Japanese 正則化を適用する行列 (dim × dim) \else Matrix to apply regularization (dim × dim) \endif
     *
     * @param dim
     * \if Japanese 行列の次元 \else Dimension of the matrix \endif
     *
     * @param lambda
     * \if Japanese 正則化パラメータ \else Regularization parameter \endif
     *
     * @param penalize_bias
     * \if Japanese バイアス項を正則化するか（false の場合、最後の対角成分は変更しない） \else Whether to regularize the bias term (if false, the last diagonal element is not modified) \endif
     */
    void applyRegularization(double *A, int dim, double lambda, bool penalize_bias);
};

#endif // RIDGE_REGRESSION_HPP
