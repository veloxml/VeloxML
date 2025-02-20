#pragma once  // 誤字修正

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <type_traits>

namespace py = pybind11;

// 補助的な trait: まずは cv 修飾子や参照を取り除いた型に対して判定を行う
template <typename T>
struct is_py_array_impl : std::false_type {};

// py::array_t は2つのテンプレートパラメータを持つので、両方に対応する部分特殊化
template <typename T, int Extra>
struct is_py_array_impl<py::array_t<T, Extra>> : std::true_type {};

// 最終的な trait: 入力型から cv 修飾子や参照を除去してから判定
template <typename T>
struct is_py_array : is_py_array_impl<std::decay_t<T>> {};

// ---- 関数テンプレートの宣言 ----
template <typename T>
std::vector<std::vector<T>> array_to_vector(const py::array_t<T>& input);  // const & に修正

template <typename T>
py::array_t<T> vector_to_array(const std::vector<std::vector<T>>& input);  // const & に修正

template <typename Array>
size_t get_num_rows(const Array& arr);

template <typename Array>
double get_element(const Array& arr, size_t row, size_t col);

template <typename Array>
std::vector<double> get_row(const Array &arr, size_t row);

template <typename Array>
size_t get_num_cols(const Array& arr);

template <typename Array>
bool is_two_dimensional(const Array& arr);

std::vector<std::vector<double>> array_to_vector(const py::array_t<double> &input);

py::array_t<double> vector_to_array(const std::vector<std::vector<double>> &input);


/**
 * @brief Col-Major順に格納された1次元配列を、行数 rows、列数 cols の2次元配列（row-major）に変換する
 * 
 * @param flat Col-Major順の1次元配列。要素は flat[j * rows + i] により、行 i, 列 j の要素が得られる
 * @param rows 行数（サンプル数）
 * @param cols 列数（クラス数など）
 * @return std::vector<std::vector<double>> 各内部 vector が1行分を表す2次元配列
 */
std::vector<std::vector<double>> convertColMajorTo2D(const std::vector<double>& flat, std::size_t rows, std::size_t cols);
// 2次元行列（std::vector<std::vector<double>>）をCol-Majorの1次元配列に変換する補助関数
std::vector<double> flattenMatrix(const std::vector<std::vector<double>>& mat);