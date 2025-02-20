#ifndef UNSUPERVISED_ESTIMATOR_BASE_H
#define UNSUPERVISED_ESTIMATOR_BASE_H

#include "base/estimator_base.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

/**
 * @class UnsupervisedEstimatorBase
 * @brief
 * \if Japanese
 * 教師なし学習モデルの基底クラス
 *
 * 本クラスは教師なし学習モデルの基底クラスであり、学習 (`fit`)、予測 (`predict`)、変換 (`transform`) などの
 * 純粋仮想関数を提供する。
 * C++ APIとPython APIの両方をサポートする。
 * \else
 * Base class for unsupervised learning models
 *
 * This class serves as a base class for unsupervised learning models and provides
 * pure virtual functions such as training (`fit`), prediction (`predict`), and transformation (`transform`).
 * Supports both C++ API and Python API.
 * \endif
 */
class UnsupervisedEstimatorBase : public EstimatorBase
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
  virtual ~UnsupervisedEstimatorBase();

  /**
   * @brief
   * \if Japanese
   * 教師なし学習モデルの学習を行う
   *
   * 教師なし学習のため `Y` は無視される。
   * \else
   * Train the unsupervised learning model
   *
   * Since it is unsupervised learning, `Y` is ignored.
   * \endif
   */
  void fit(const std::vector<double> &X, const std::vector<double> &Y, std::size_t n, std::size_t m) override
  {
    fit(X, n, m);
  }

  /**
   * @brief
   * \if Japanese
   * 教師なし学習モデルの学習を行う（純粋仮想関数）
   * \else
   * Train the unsupervised learning model (pure virtual function)
   * \endif
   */
  virtual void fit(const std::vector<double> &X, std::size_t n, std::size_t m) = 0;

  /**
   * @brief
   * \if Japanese
   * クラスタリングなどの予測を行う
   * \else
   * Perform clustering or other predictions
   * \endif
   */
  virtual std::vector<double> predict(const std::vector<double> &X, std::size_t n, std::size_t m) = 0;

  /**
   * @brief
   * \if Japanese
   * 特徴変換を行う
   * \else
   * Perform feature transformation
   * \endif
   */
  virtual std::vector<double> transform(const std::vector<double> &X, std::size_t n, std::size_t m) = 0;

  /**
   * @brief
   * \if Japanese
   * 学習後に特徴変換を行う
   * \else
   * Train and then perform feature transformation
   * \endif
   */
  virtual std::vector<double> fit_transform(const std::vector<double> &X, std::size_t n, std::size_t m)
  {
    fit(X, n, m);
    return transform(X, n, m);
  }

  /**
   * @brief
   * \if Japanese
   * 学習後に予測を行う
   * \else
   * Train and then perform prediction
   * \endif
   */
  virtual std::vector<double> fit_predict(const std::vector<double> &X, std::size_t n, std::size_t m)
  {
    fit(X, n, m);
    return predict(X, n, m);
  }

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
   */
  void fit_py(const py::array_t<double> &X)
  {
    auto bufX = X.request();
    if (bufX.ndim != 2)
      throw std::runtime_error("fit: X must be a 2-dimensional array");

    // pybind11::ssize_t → std::size_t への変換（データコピーには問題なし）
    std::size_t n = static_cast<std::size_t>(bufX.shape[0]);
    std::size_t m = static_cast<std::size_t>(bufX.shape[1]);

    std::vector<double> vecX(static_cast<double *>(bufX.ptr),
                             static_cast<double *>(bufX.ptr) + n * m);
    fit(vecX, n, m);
  }

  /**
   * @brief
   * \if Japanese
   * Python API用の `transform` メソッド
   *
   * Pythonの `numpy.ndarray` からデータを取得し、C++ の `transform` メソッドを呼び出す。
   * \else
   * `transform` method for Python API
   *
   * Retrieves data from Python's `numpy.ndarray` and calls the C++ `transform` method.
   * \endif
   */
  py::array_t<double> transform_py(const py::array_t<double> &X)
  {
    auto bufX = X.request();
    if (bufX.ndim != 2)
      throw std::runtime_error("transform: X must be a 2-dimensional array");

    std::size_t n = static_cast<std::size_t>(bufX.shape[0]);
    std::size_t m = static_cast<std::size_t>(bufX.shape[1]);

    std::vector<double> vecX(static_cast<double *>(bufX.ptr),
                             static_cast<double *>(bufX.ptr) + n * m);
    std::vector<double> result = transform(vecX, n, m);

    // 形状指定時に py::ssize_t へキャストして narrowing 警告を回避
    return py::array_t<double>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(m)},
                               result.data());
  }

  /**
   * @brief
   * \if Japanese
   * Python API用の `fit_transform` メソッド
   *
   * Pythonの `numpy.ndarray` からデータを取得し、C++ の `fit_transform` メソッドを呼び出す。
   * \else
   * `fit_transform` method for Python API
   *
   * Retrieves data from Python's `numpy.ndarray` and calls the C++ `fit_transform` method.
   * \endif
   */
  py::array_t<double> fit_transform_py(const py::array_t<double> &X)
  {
    auto bufX = X.request();
    if (bufX.ndim != 2)
      throw std::runtime_error("fit_transform: X must be a 2-dimensional array");

    std::size_t n = static_cast<std::size_t>(bufX.shape[0]);
    std::size_t m = static_cast<std::size_t>(bufX.shape[1]);

    std::vector<double> vecX(static_cast<double *>(bufX.ptr),
                             static_cast<double *>(bufX.ptr) + n * m);
    std::vector<double> result = fit_transform(vecX, n, m);

    return py::array_t<double>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(m)},
                               result.data());
  }

  /**
   * @brief
   * \if Japanese
   * Python API用の `fit_predict` メソッド
   *
   * Pythonの `numpy.ndarray` からデータを取得し、C++ の `fit_predict` メソッドを呼び出す。
   * \else
   * `fit_predict` method for Python API
   *
   * Retrieves data from Python's `numpy.ndarray` and calls the C++ `fit_predict` method.
   * \endif
   */
  py::array_t<double> fit_predict_py(const py::array_t<double> &X)
  {
    auto bufX = X.request();
    if (bufX.ndim != 2)
      throw std::runtime_error("fit_predict: X must be a 2-dimensional array");

    std::size_t n = static_cast<std::size_t>(bufX.shape[0]);
    std::size_t m = static_cast<std::size_t>(bufX.shape[1]);

    std::vector<double> vecX(static_cast<double *>(bufX.ptr),
                             static_cast<double *>(bufX.ptr) + n * m);
    std::vector<double> result = fit_predict(vecX, n, m);

    // ここでは出力が1次元配列と仮定
    return py::array_t<double>({static_cast<py::ssize_t>(n)},
                               result.data());
  }
};

#endif // UNSUPERVISED_ESTIMATOR_BASE_H
