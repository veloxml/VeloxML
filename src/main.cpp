// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <cmath>
// #include "linear/linear_regression.hpp" // CPU版
// #include "metal/linear/cxx_linear_regression_metal.h" // Metal版

// // MSE計算
// double mean_squared_error(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
//     double mse = 0.0;
//     int n = static_cast<int>(y_true.size());
//     for (int i = 0; i < n; i++) {
//         double diff = y_true[i] - y_pred[i];
//         mse += diff * diff;
//     }
//     return mse / n;
// }

// int main() {
//     int rows = 1000; // サンプル数
//     int cols = 1; // 特徴量の数（1次元）

//     // データセット作成 (y = 3x + 2)
//     std::vector<double> X(rows);
//     std::vector<double> y(rows);
//     for (int i = 0; i < rows; i++) {
//         X[i] = static_cast<double>(i) / 10.0; // 0.0, 0.1, ..., 99.9
//         y[i] = 3.0 * X[i] + 2.0;
//     }

//     // CPU版で学習
//     LinearRegression cpu_model;
//     auto start_cpu = std::chrono::high_resolution_clock::now();
//     cpu_model.fit(X, y, rows, cols);
//     auto end_cpu = std::chrono::high_resolution_clock::now();
//     auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
//     std::cout << "CPU Training Time: " << duration_cpu.count() << " ms" << std::endl;

//     // Metal版で学習
//     LinearRegressionMetal metal_model;
//     auto start_metal = std::chrono::high_resolution_clock::now();
//     metal_model.fit(X, y, rows, cols);
//     auto end_metal = std::chrono::high_resolution_clock::now();
//     auto duration_metal = std::chrono::duration_cast<std::chrono::milliseconds>(end_metal - start_metal);
//     std::cout << "Metal Training Time: " << duration_metal.count() << " ms" << std::endl;

//     // 予測結果の取得
//     std::vector<double> cpu_pred = cpu_model.predict(X, rows, cols);

//     std::vector<double> metal_pred = metal_model.predict(X, rows, cols);

//     // MSE計算
//     double mse_cpu = mean_squared_error(y, cpu_pred);
//     double mse_metal = mean_squared_error(y, metal_pred);

//     std::cout << "CPU MSE: " << mse_cpu << std::endl;
//     std::cout << "Metal MSE: " << mse_metal << std::endl;

//     return 0;
// }
