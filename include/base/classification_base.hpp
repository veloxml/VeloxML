/**
* @file classification_base.hpp
* @brief A file that defines the base class for classifiers
* @author Yuji Chinen
* @date 2024/02/19
*
* @details This file defines a base class for an inferer that trains classification tasks.
* @note 
*/

#ifndef CLASSIFICATION_BASE_H
#define CLASSIFICATION_BASE_H

#include <base/estimator_base.hpp>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

/**
 * @class ClassificationBase
 * @brief
 * \if Japanese
 * 分類モデルの基底クラス
 *
 * 本クラスは分類モデルの基底クラスとして、学習 (`fit`)、予測 (`predict`)、確率予測 (`predict_proba`) の
 * 純粋仮想関数を提供する。
 * C++ APIとPython APIの両方をサポートする。
 * \else
 * Base class for classification models
 *
 * This class serves as a base class for classification models and provides
 * pure virtual functions for training (`fit`), prediction (`predict`), and probability prediction (`predict_proba`).
 * Supports both C++ API and Python API.
 * \endif
 */
class ClassificationBase : public EstimatorBase
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
    virtual ~ClassificationBase();

    /**
     * @brief
     * \if Japanese
     * モデルを学習する
     *
     * 説明変数 `X` と目的変数 `Y` を用いて分類モデルを学習する。
     * \else
     * Train the classification model
     *
     * Trains the classification model using feature variable `X` and target variable `Y`.
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
     * 出力データ（クラスラベル）のベクトル (サイズ: `n`)
     * \else
     * Output data (class labels) as a vector (size: `n`)
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
     */
    virtual void fit(const std::vector<double> &X,
                     const std::vector<double> &Y,
                     std::size_t n, std::size_t m) = 0;

    /**
     * @brief
     * \if Japanese
     * クラスラベルの予測を行う
     *
     * 入力データ `X` に対して分類を行い、クラスラベルを予測する。
     * \else
     * Perform class label prediction
     *
     * Classifies the given input data `X` and predicts the class labels.
     * \endif
     *
     * @param X
     * \if Japanese
     * 入力データのCol-Major配列 (サイズ: `l × m`)
     * \else
     * Input data as a Col-Major array (size: `l × m`)
     * \endif
     *
     * @param l
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
     * 予測されたクラスラベルのベクトル (サイズ: `l`)
     * \else
     * Predicted class labels as a vector (size: `l`)
     * \endif
     */
    virtual std::vector<double> predict(const std::vector<double> &X,
                                        std::size_t l, std::size_t m) = 0;

    /**
     * @brief
     * \if Japanese
     * クラスごとの確率予測を行う
     *
     * 入力データ `X` に対して、各クラスの確率を予測する。
     * \else
     * Perform class probability prediction
     *
     * Predicts the probabilities for each class given the input data `X`.
     * \endif
     *
     * @param X
     * \if Japanese
     * 入力データのCol-Major配列 (サイズ: `l × m`)
     * \else
     * Input data as a Col-Major array (size: `l × m`)
     * \endif
     *
     * @param l
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
     * クラスごとの確率を含むCol-Major配列 (サイズ: `l × num_classes`)
     * \else
     * Col-Major array containing probabilities for each class (size: `l × num_classes`)
     * \endif
     */
    virtual std::vector<double> predict_proba(const std::vector<double> &X,
                                              std::size_t l, std::size_t m) = 0;

    /**
     * @brief
     * \if Japanese
     * Python API用の `predict_proba` メソッド
     *
     * Pythonの `numpy.ndarray` からデータを取得し、C++ の `predict_proba` メソッドを呼び出す。
     * \else
     * `predict_proba` method for Python API
     *
     * Retrieves data from Python's `numpy.ndarray` and calls the C++ `predict_proba` method.
     * \endif
     *
     * @param X
     * \if Japanese
     * 入力データのNumPy配列（2次元）
     * \else
     * Input data NumPy array (2D)
     * \endif
     *
     * @return
     * \if Japanese
     * クラスごとの確率を含むNumPy配列 (サイズ: `l × num_classes`)
     * \else
     * NumPy array containing probabilities for each class (size: `l × num_classes`)
     * \endif
     *
     * @throws std::runtime_error
     * \if Japanese
     * Xが2次元配列でない場合
     * \else
     * If X is not a 2D array
     * \endif
     */
    py::array_t<double> predict_proba_py(const py::array_t<double> &X)
    {
        auto bufX = X.request();
        if (bufX.ndim != 2)
            throw std::runtime_error("predict_proba: X must be a 2-dimensional array");

        std::size_t l = bufX.shape[0];
        std::size_t m = bufX.shape[1];

        std::vector<double> vecX(static_cast<double *>(bufX.ptr), static_cast<double *>(bufX.ptr) + l * m);
        std::vector<double> result = predict_proba(vecX, l, m);

        std::size_t num_classes = result.size() / l;
        if (num_classes == 0)
            throw std::runtime_error("predict_proba: Invalid output size");

        return py::array_t<double>({l, num_classes}, result.data());
    }
};

#endif // CLASSIFICATION_BASE_H
