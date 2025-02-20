/**
* @file estimator_base.hpp
* @brief A file that defines the base class for the inference engine
* @author Yuji Chinen
* @date 2024/02/19
*
* @details This file defines a base class for an inferer that trains classification tasks.
* @note 
*/

#ifndef ESTIMATOR_BASE_H
#define ESTIMATOR_BASE_H

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

/**
 * @class EstimatorBase
 * @brief
 * \if Japanese
 * 機械学習モデルの基底クラス。
 *
 * このクラスは機械学習モデルの基底クラスとして、学習（fit）と予測（predict）を行う純粋仮想関数を提供する。
 * C++ APIとPython APIの両方をサポートする。
 * \else
 * Base class for machine learning models.
 *
 * This class acts as a base class for machine learning models and provides
 * pure virtual functions for training (fit) and prediction (predict).
 * Supports both C++ API and Python API.
 * \endif
 */
class EstimatorBase
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
  virtual ~EstimatorBase();

  /**
   * @brief
   * \if Japanese
   * モデルを学習する
   * \else
   * Train the model
   * \endif
   *
   * @param X
   * \if Japanese
   * 説明変数のベクトル（サイズ: n × m）
   * \else
   * Feature vector (size: n × m)
   * \endif
   *
   * @param Y
   * \if Japanese
   * 目的変数のベクトル（サイズ: n）
   * \else
   * Target variable vector (size: n)
   * \endif
   *
   * @param n
   * \if Japanese
   * サンプル数
   * \else
   * Number of samples
   * \endif
   *
   * @param m
   * \if Japanese
   * 特徴量の次元数
   * \else
   * Number of features
   * \endif
   *
   * @throws std::runtime_error
   * \if Japanese
   * XまたはYのサイズが不正な場合
   * \else
   * If X or Y has an invalid size
   * \endif
   */
  virtual void fit(
      const std::vector<double> &X,
      const std::vector<double> &Y,
      std::size_t n, std::size_t m) = 0;

  /**
   * @brief
   * \if Japanese
   * 予測を行う
   * \else
   * Perform prediction
   * \endif
   *
   * @param X
   * \if Japanese
   * 入力データのベクトル（サイズ: l × m）
   * \else
   * Input data vector (size: l × m)
   * \endif
   *
   * @param l
   * \if Japanese
   * サンプル数
   * \else
   * Number of samples
   * \endif
   *
   * @param m
   * \if Japanese
   * 特徴量の次元数
   * \else
   * Number of features
   * \endif
   *
   * @return
   * \if Japanese
   * 予測結果のベクトル（サイズ: l）
   * \else
   * Predicted result vector (size: l)
   * \endif
   */
  virtual std::vector<double> predict(const std::vector<double> &X,
                                      std::size_t l, std::size_t m) = 0;

  /**
   * @brief
   * \if Japanese
   * Python API用の `fit` メソッド
   *
   * Pythonの `numpy.ndarray` からデータを取得し、C++ の `fit` メソッドを呼び出す。
   * \else
   * `fit` method for Python API
   *
   * Retrieves data from Python's `numpy.ndarray` and calls the C++ `fit` method.
   * \endif
   *
   * @param X
   * \if Japanese
   * 説明変数のNumPy配列（2次元）
   * \else
   * Feature variable NumPy array (2D)
   * \endif
   *
   * @param Y
   * \if Japanese
   * 目的変数のNumPy配列（1次元）
   * \else
   * Target variable NumPy array (1D)
   * \endif
   *
   * @throws std::runtime_error
   * \if Japanese
   * Xが2次元配列でない場合
   * \else
   * If X is not a 2D array
   * \endif
   */
  void fit_py(const py::array_t<double> &X, const py::array_t<double> &Y)
  {
    auto bufX = X.request();
    auto bufY = Y.request();

    if (bufX.ndim != 2)
      throw std::runtime_error("fit: X must be a 2-dimensional array");

    std::size_t n = bufX.shape[0];
    std::size_t m = bufX.shape[1];

    std::vector<double> vecX(static_cast<double *>(bufX.ptr), static_cast<double *>(bufX.ptr) + n * m);
    std::vector<double> vecY(static_cast<double *>(bufY.ptr), static_cast<double *>(bufY.ptr) + n);

    fit(vecX, vecY, n, m);
  }

  /**
   * @brief
   * \if Japanese
   * Python API用の `predict` メソッド
   *
   * Pythonの `numpy.ndarray` からデータを取得し、C++ の `predict` メソッドを呼び出す。
   * \else
   * `predict` method for Python API
   *
   * Retrieves data from Python's `numpy.ndarray` and calls the C++ `predict` method.
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
   * 予測結果のNumPy配列（1次元）
   * \else
   * Predicted result NumPy array (1D)
   * \endif
   *
   * @throws std::runtime_error
   * \if Japanese
   * Xが2次元配列でない場合
   * \else
   * If X is not a 2D array
   * \endif
   */
  py::array_t<double> predict_py(const py::array_t<double> &X)
  {
    auto bufX = X.request();
    if (bufX.ndim != 2)
      throw std::runtime_error("predict: X must be a 2-dimensional array");

    std::size_t l = bufX.shape[0];
    std::size_t m = bufX.shape[1];

    std::vector<double> vecX(static_cast<double *>(bufX.ptr), static_cast<double *>(bufX.ptr) + l * m);
    std::vector<double> result = predict(vecX, l, m);

    return py::array_t<double>(result.size(), result.data());
  }
};

#endif // ESTIMATOR_BASE_H
