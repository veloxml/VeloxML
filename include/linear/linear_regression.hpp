#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "base/regression_base.hpp"
#include <vector>
#include <stdexcept>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tbb/parallel_for.h>

/**
 * @enum LinearDecompositionMode
 * @brief
 * \if Japanese
 * 線形回帰の解法を表す列挙型
 *
 * LU分解、QR分解、SVD分解のいずれかを選択可能。
 * \else
 * Enumeration representing decomposition methods for linear regression
 *
 * Supports LU decomposition, QR decomposition, and SVD decomposition.
 * \endif
 */
enum class LinearDecompositionMode
{
    LU, ///< \if Japanese Cholesky 分解を利用（実装上は LU として扱う） \else Uses Cholesky decomposition (handled as LU in implementation) \endif
    QR, ///< \if Japanese QR 分解を利用 \else Uses QR decomposition \endif
    SVD ///< \if Japanese SVD 分解を利用 \else Uses SVD decomposition \endif
};

/**
 * @class LinearRegression
 * @brief
 * \if Japanese
 * 線形回帰の実装クラス
 *
 * 内部でバイアス項を付加して学習を行い、LU/QR/SVD による解法を選択可能。
 * 入出力は1次元の Col-Major 配列としてやり取りし、行数・列数は各関数の引数で指定する。
 * 内部で再利用可能なバッファを持ち、メモリアロケーションを削減する。
 * \else
 * Implementation class for linear regression
 *
 * Trains with an internally added bias term and supports LU/QR/SVD decomposition methods.
 * Inputs and outputs are handled as 1D Col-Major arrays, with row and column sizes specified via function arguments.
 * Uses an internal reusable buffer to reduce memory allocations.
 * \endif
 */
class LinearRegression : public RegressionBase
{
public:
    /**
     * @brief
     * \if Japanese
     * 線形回帰モデルのコンストラクタ
     *
     * 分解モードを指定して線形回帰モデルを初期化する。
     * \else
     * Constructor for the linear regression model
     *
     * Initializes the linear regression model with the specified decomposition mode.
     * \endif
     *
     * @param mode
     * \if Japanese 分解モード（デフォルト: LU） \else Decomposition mode (default: LU) \endif
     */
    LinearRegression(LinearDecompositionMode mode = LinearDecompositionMode::LU);

    /**
     * @brief
     * \if Japanese
     * デストラクタ
     * \else
     * Destructor
     * \endif
     */
    ~LinearRegression();

    // --- C++ API ---
    /**
     * @brief
     * \if Japanese
     * モデルを学習する
     *
     * 説明変数 `X` (n × m) と目的変数 `Y` (n × 1) を用いて線形回帰モデルを学習する。
     * \else
     * Train the model
     *
     * Trains the linear regression model using feature variables `X` (n × m) and target variables `Y` (n × 1).
     * \endif
     */
    void fit(const std::vector<double> &X, const std::vector<double> &Y, std::size_t n, std::size_t m) override;

    /**
     * @brief
     * \if Japanese
     * モデルによる予測を行う
     *
     * 入力データ `X` (n × m) に対して線形回帰の予測を行い、出力として (n × 1) のベクトルを返す。
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
    LinearDecompositionMode mode_; ///< \if Japanese 分解モード（LU, QR, SVD） \else Decomposition mode (LU, QR, SVD) \endif
    std::vector<double> weights_;  ///< \if Japanese 学習済みの重み \else Learned weights \endif
    double bias_;                  ///< \if Japanese 学習済みのバイアス項 \else Learned bias term \endif

    mutable std::vector<double> X_aug_buffer_; ///< \if Japanese メモリアロケーションを削減するための内部バッファ \else Internal buffer to reduce memory allocation \endif

    /**
     * @brief
     * \if Japanese
     * 入力行列 `X` にバイアス項を追加する
     *
     * `X` (n × m) にバイアス項 (1.0 の列) を付加し、(n × (m+1)) のCol-Major配列として返す。
     * 内部バッファを活用し、メモリアロケーションを削減する。
     * \else
     * Add a bias term to the input matrix `X`
     *
     * Appends a bias term (a column of 1.0) to `X` (n × m) and returns a Col-Major array of size (n × (m+1)).
     * Uses an internal buffer to reduce memory allocation.
     * \endif
     */
    const std::vector<double> &augmentWithBias(const double *X, int n, int m);

    /**
     * @brief
     * \if Japanese
     * 回帰パラメータを計算する
     *
     * LU, QR, SVD のいずれかの手法を用いて、正規方程式を解く。
     * \else
     * Compute regression parameters
     *
     * Solves the normal equations using LU, QR, or SVD decomposition.
     * \endif
     *
     * @param X
     * \if Japanese 元の行列 (n × m) のCol-Major配列 \else Original matrix (n × m) as a Col-Major array \endif
     *
     * @param n
     * \if Japanese 行数 \else Number of rows \endif
     *
     * @param m
     * \if Japanese 特徴量数 \else Number of features \endif
     *
     * @param Y
     * \if Japanese 目的変数 (n × 1) の配列 \else Target variable array (n × 1) \endif
     */
    void computeRegression(const double *X, int n, int m, const double *Y);
};

#endif // LINEAR_REGRESSION_HPP
