#include "utils/utilities.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

// 2次元の py::array_t を std::vector<std::vector<T>> に変換
template <typename T>
std::vector<std::vector<T>> array_to_vector(const py::array_t<T> &input)
{
  // 2次元配列としてアクセス（次元が異なる場合は例外が発生します）
  auto buf = input.template unchecked<2>();
  ssize_t rows = buf.shape(0);
  ssize_t cols = buf.shape(1);

  std::vector<std::vector<T>> output(rows, std::vector<T>(cols));
  for (ssize_t i = 0; i < rows; i++)
  {
    for (ssize_t j = 0; j < cols; j++)
    {
      output[i][j] = buf(i, j);
    }
  }
  return output;
}

// std::vector<std::vector<T>> を 2次元の py::array_t に変換
template <typename T>
py::array_t<T> vector_to_array(const std::vector<std::vector<T>> &input)
{
  if (input.empty())
    return py::array_t<T>(); // 空の場合は空の配列を返す

  ssize_t rows = input.size();
  ssize_t cols = input[0].size();

  // 新たに配列を生成（この場合、pybind11がメモリを管理する）
  py::array_t<T> output({rows, cols});
  auto buf = output.template mutable_unchecked<2>();

  for (ssize_t i = 0; i < rows; i++)
  {
    // 各行のサイズが同じであることを前提としています
    for (ssize_t j = 0; j < cols; j++)
    {
      buf(i, j) = input[i][j];
    }
  }
  return output;
}

// 行数を取得する共通インターフェース
template <typename Array>
size_t get_num_rows(const Array &arr)
{
  if constexpr (is_py_array<Array>::value)
  {
    // py::array_tの場合は request() でバッファ情報を取得
    auto buf = arr.request();
    return buf.shape[0];
  }
  else
  {
    return arr.size();
  }
}

// ----- 列数を取得する関数 -----
// 2次元配列でなければ例外を投げ、そうであれば
// py::array_t ならば buf.shape[1]、std::vector<std::vector<T>> なら最初の行のサイズを返します。
template <typename Array>
size_t get_num_cols(const Array &arr)
{
  if (!is_two_dimensional(arr))
  {
    throw std::runtime_error("Input is not a two-dimensional array.");
  }

  if constexpr (is_py_array<Array>::value)
  {
    auto buf = arr.request();
    return buf.shape[1];
  }
  else
  {
    return arr[0].size();
  }
}

// ----- 2次元配列かどうかの判定関数 -----
// py::array_t ではバッファ情報の ndim を、std::vector<std::vector<T>> では
// 空でないことと最初の要素が空でないことを確認することで判定しています。
template <typename Array>
bool is_two_dimensional(const Array &arr)
{
  if constexpr (is_py_array<Array>::value)
  {
    auto buf = arr.request();
    return (buf.ndim == 2);
  }
  else
  {
    // std::vector<std::vector<T>> として扱う場合
    return !arr.empty() && !arr[0].empty();
  }
}

// 指定された行・列の要素を取得する共通インターフェース
template <typename Array>
double get_element(const Array &arr, size_t row, size_t col)
{
  if constexpr (is_py_array<Array>::value)
  {
    auto buf = arr.request();
    // ここでは double 型を想定。必要に応じてテンプレート引数などで対応してください。
    auto *ptr = static_cast<double *>(buf.ptr);
    return ptr[row * buf.shape[1] + col];
  }
  else
  {
    return arr[row][col];
  }
}

// 一行を取得するヘルパー関数
template <typename Array>
std::vector<double> get_row(const Array &arr, size_t row) {
    if constexpr (is_py_array<Array>::value) {
        auto buf = arr.request();
        size_t cols = buf.shape[1];  // 列数
        auto *ptr = static_cast<double *>(buf.ptr);

        std::vector<double> row_data(cols);
        for (size_t j = 0; j < cols; ++j) {
            row_data[j] = ptr[row * cols + j];
        }
        return row_data;
    } else {
        return arr[row];  // std::vector<std::vector<double>> の場合はそのまま返す
    }
}

namespace py = pybind11;

// ----- py::array_t<double> → std::vector<std::vector<double>> 変換 -----
std::vector<std::vector<double>> array_to_vector(const py::array_t<double> &input) {
    auto buf = input.unchecked<2>();  // 高速アクセス用
    size_t rows = buf.shape(0);
    size_t cols = buf.shape(1);

    std::vector<std::vector<double>> output(rows, std::vector<double>(cols));

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            output[i][j] = buf(i, j);
        }
    }
    return output;
}

// ----- std::vector<std::vector<double>> → py::array_t<double> 変換 -----
py::array_t<double> vector_to_array(const std::vector<std::vector<double>> &input) {
    if (input.empty() || input[0].empty()) {
        return py::array_t<double>();  // 空の入力に対応
    }

    size_t rows = input.size();
    size_t cols = input[0].size();

    // pybind11がメモリを管理するようにnumpy配列を作成
    py::array_t<double> output({rows, cols});
    auto buf = output.mutable_unchecked<2>();  // 高速書き込み用

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            buf(i, j) = input[i][j];
        }
    }
    return output;
}

/**
 * @brief Col-Major順に格納された1次元配列を、行数 rows、列数 cols の2次元配列（row-major）に変換する
 * 
 * @param flat Col-Major順の1次元配列。要素は flat[j * rows + i] により、行 i, 列 j の要素が得られる
 * @param rows 行数（サンプル数）
 * @param cols 列数（クラス数など）
 * @return std::vector<std::vector<double>> 各内部 vector が1行分を表す2次元配列
 */
std::vector<std::vector<double>> convertColMajorTo2D(const std::vector<double>& flat, std::size_t rows, std::size_t cols)
{
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            // Col-Major 配列では、(i,j)成分は flat[j * rows + i] に格納されている
            result[i][j] = flat[j * rows + i];
        }
    }
    return result;
}

// 2次元行列（std::vector<std::vector<double>>）をCol-Majorの1次元配列に変換する補助関数
std::vector<double> flattenMatrix(const std::vector<std::vector<double>>& mat)
{
    if (mat.empty())
        return {};

    size_t n = mat.size();          // サンプル数（行数）
    size_t m = mat[0].size();         // 特徴数（列数）
    std::vector<double> flat(n * m, 0.0);
    // Col-Major順: 各列毎に連続する形で格納
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            flat[j * n + i] = mat[i][j];
        }
    }
    return flat;
}
