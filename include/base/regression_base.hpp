#ifndef REGRESSION_BASE_HPP
#define REGRESSION_BASE_HPP

#include <base/estimator_base.hpp>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

/**
 * @class RegressionBase
 * @brief 
 * \if Japanese
 * 回帰モデルの基底クラス
 * 
 * 本クラスは回帰モデルの基底クラスであり、学習 (`fit`) および予測 (`predict`) のための
 * 純粋仮想関数を提供する。  
 * 入出力は一次元のCol-Major配列でやり取りし、行列の形状（行数・列数）は関数引数で指定する。
 * \else
 * Base class for regression models
 * 
 * This class serves as a base class for regression models and provides pure
 * virtual functions for training (`fit`) and prediction (`predict`).  
 * Input and output are exchanged as 1D Col-Major arrays, and the matrix
 * shape (number of rows and columns) is specified via function arguments.
 * \endif
 */
class RegressionBase : public EstimatorBase
{
public:
    /**
     * @brief 
     * \if Japanese
     * 仮想デストラクタ
     * \else
     * Virtual destructor
     * \endif
     */
    virtual ~RegressionBase();

    // --- C++ API ---

    /**
     * @brief 
     * \if Japanese
     * モデルを学習する
     * 
     * 説明変数 `X` と目的変数 `Y` を用いてモデルを学習する。
     * \else
     * Train the model
     * 
     * Trains the model using the feature variable `X` and target variable `Y`.
     * \endif
     * 
     * @param X 
     * \if Japanese
     * 入力データ（説明変数）のCol-Major配列 (サイズ: `n × m`)
     * \else
     * Input data (feature variable) as a Col-Major array (size: `n × m`)
     * \endif
     * 
     * @param Y 
     * \if Japanese
     * 出力データ（目的変数）のCol-Major配列 (サイズ: `n × p`, 通常 `p=1`)
     * \else
     * Output data (target variable) as a Col-Major array (size: `n × p`, usually `p=1`)
     * \endif
     * 
     * @param n 
     * \if Japanese
     * サンプル数（行数）
     * \else
     * Number of samples (rows)
     * \endif
     * 
     * @param m 
     * \if Japanese
     * 特徴量の次元数（列数）
     * \else
     * Number of features (columns)
     * \endif
     * 
     * @throws std::runtime_error 
     * \if Japanese
     * X または Y のサイズが不正な場合
     * \else
     * If the size of X or Y is invalid
     * \endif
     */
    virtual void fit(const std::vector<double> &X, const std::vector<double> &Y, std::size_t n, std::size_t m) = 0;

    /**
     * @brief 
     * \if Japanese
     * モデルによる予測を行う
     * 
     * 入力データ `X` に対して予測を行い、予測結果を返す。
     * \else
     * Perform prediction using the trained model
     * 
     * Predicts output for the given input data `X` and returns the predicted result.
     * \endif
     * 
     * @param X 
     * \if Japanese
     * 入力データのCol-Major配列 (サイズ: `n × m`)
     * \else
     * Input data as a Col-Major array (size: `n × m`)
     * \endif
     * 
     * @param n 
     * \if Japanese
     * サンプル数（行数）
     * \else
     * Number of samples (rows)
     * \endif
     * 
     * @param m 
     * \if Japanese
     * 特徴量の次元数（列数）
     * \else
     * Number of features (columns)
     * \endif
     * 
     * @return 
     * \if Japanese
     * 予測結果のCol-Major配列 (サイズ: `n × p`, 通常 `p=1`)
     * \else
     * Predicted result as a Col-Major array (size: `n × p`, usually `p=1`)
     * \endif
     */
    virtual std::vector<double> predict(const std::vector<double> &X, std::size_t n, std::size_t m) = 0;
};

#endif // REGRESSION_BASE_HPP
