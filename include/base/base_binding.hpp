/**
* @file base_binding.hpp
* @brief A file for base class bindings
* @author Yuji Chinen
* @date 2024/02/19
*
* @details This file defines helper functions used when binding base classes with pybind11.
* @note 
*/

#ifndef BASE_BINDING_HPP
#define BASE_BINDING_HPP

#include <pybind11/pybind11.h>
#include <memory>
#include "base/unsupervised_base.hpp"
#include "base/classification_base.hpp"
#include "base/regression_base.hpp"
#include "base/estimator_base.hpp"

namespace py = pybind11;

/**
 * @brief
 * \if Japanese
 * EstimatorBase クラスを Python にバインドする
 *
 * Pybind11 を使用して、C++ の `EstimatorBase` クラスを Python で利用可能にする。
 * \else
 * Bind the EstimatorBase class to Python
 *
 * Uses Pybind11 to make the C++ `EstimatorBase` class available in Python.
 * \endif
 *
 * @param m
 * \if Japanese
 * Pybind11 のモジュールオブジェクト
 * \else
 * Pybind11 module object
 * \endif
 */
void bind_estimator_base(py::module_ &m);

/**
 * @brief
 * \if Japanese
 * RegressionBase クラスを Python にバインドする
 *
 * Pybind11 を使用して、C++ の `RegressionBase` クラスを Python で利用可能にする。
 * \else
 * Bind the RegressionBase class to Python
 *
 * Uses Pybind11 to make the C++ `RegressionBase` class available in Python.
 * \endif
 *
 * @param m
 * \if Japanese
 * Pybind11 のモジュールオブジェクト
 * \else
 * Pybind11 module object
 * \endif
 */
void bind_regression_base(py::module_ &m);

/**
 * @brief
 * \if Japanese
 * ClassificationBase クラスを Python にバインドする
 *
 * Pybind11 を使用して、C++ の `ClassificationBase` クラスを Python で利用可能にする。
 * \else
 * Bind the ClassificationBase class to Python
 *
 * Uses Pybind11 to make the C++ `ClassificationBase` class available in Python.
 * \endif
 *
 * @param m
 * \if Japanese
 * Pybind11 のモジュールオブジェクト
 * \else
 * Pybind11 module object
 * \endif
 */
void bind_classification_base(py::module_ &m);

/**
 * @brief
 * \if Japanese
 * UnsupervisedEstimatorBase クラスを Python にバインドする
 *
 * Pybind11 を使用して、C++ の `UnsupervisedEstimatorBase` クラスを Python で利用可能にする。
 * \else
 * Bind the UnsupervisedEstimatorBase class to Python
 *
 * Uses Pybind11 to make the C++ `UnsupervisedEstimatorBase` class available in Python.
 * \endif
 *
 * @param m
 * \if Japanese
 * Pybind11 のモジュールオブジェクト
 * \else
 * Pybind11 module object
 * \endif
 */
void bind_unsupervised_base(py::module_ &m);

#endif // BASE_BINDING_HPP
