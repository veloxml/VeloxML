#include "svm/svm_regression.hpp" // SVMRegression クラスのヘッダ（適宜パスを調整してください）
#include "test_data_utils.hpp"    // generate_data(M, N) を含む
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <iostream>

// 許容誤差（double の精度を考慮）
constexpr double TOLERANCE = 1e-2;

// SVM回帰のテストクラス
// パラメータ: サンプル数 M, 特徴量数 N, C, ε, カーネル種 (KernelType),
//             gamma, degree, coef0, approx_dim
class SVMRegressionTest
    : public ::testing::TestWithParam<std::tuple<
          int, int, double, double, KernelType, double, int, double, int>> {};

TEST_P(SVMRegressionTest, FitPredictTest) {
  // パラメータの取得
  int M = std::get<0>(GetParam());
  int N = std::get<1>(GetParam());
  double C = std::get<2>(GetParam());
  double epsilon = std::get<3>(GetParam());
  KernelType kernel = std::get<4>(GetParam());
  double gamma = std::get<5>(GetParam());
  int degree = std::get<6>(GetParam());
  double coef0 = std::get<7>(GetParam());
  int approx_dim = std::get<8>(GetParam());

  // サンプル数が特徴量数より少ない場合はテストをスキップ
  if (M < N) {
    GTEST_SKIP() << "Skipping test: M (" << M << ") < N (" << N << ")";
  }

  // generate_data(M, N) は (M×N) の Col‐Major 配列 X と (M×1) の配列 Y
  // を返すとする
  auto [X, Y] = generate_data_simple(M, N);
  standardize(X, M, N);

  // SVMRegression モデルの生成
  // コンストラクタ: SVMRegression(C, epsilon, tol, max_iter, kernel_type,
  // gamma, degree, coef0, approx_dim) ここでは max_iter = 1000, tol = 1e-6
  // として固定
  SVMRegression model(C, epsilon, 1e-6, 1000, kernel, gamma, degree, coef0,
                      approx_dim);

  // 学習を実行：X は (M×N) の1次元 Col‐Major 配列、Y は (M×1) の配列
  ASSERT_NO_THROW(model.fit(X, Y, static_cast<std::size_t>(M),
                            static_cast<std::size_t>(N)));

  // 予測値を取得
  std::vector<double> Y_pred;
  ASSERT_NO_THROW(Y_pred = model.predict(X, M, N));

  // 予測結果のサイズチェック（(M×1) の配列となるはず）
  ASSERT_EQ(Y_pred.size(), static_cast<size_t>(M));

  // 予測値が NaN や Inf でないかチェック
  for (int i = 0; i < M; ++i) {
    ASSERT_TRUE(std::isfinite(Y_pred[i]))
        << "Prediction at index " << i << " is non-finite";
  }

  // 予測誤差 (MSE) の評価
  double mse = 0.0;
  for (int i = 0; i < M; ++i) {
    double diff = Y_pred[i] - Y[i];
    mse += diff * diff;
  }
  mse /= M;
  ASSERT_LT(mse, TOLERANCE) << "Mean Squared Error too high: " << mse;

  // // カーネルが LINEAR または APPROX_RBF
  // // の場合、学習済みの回帰係数をチェックする
  // if (kernel == KernelType::LINEAR || kernel == KernelType::APPROX_RBF) {
  //   std::vector<double> coefficients = model.get_weights();
  //   ASSERT_EQ(coefficients.size(), static_cast<size_t>(N));
  //   for (int j = 0; j < N; ++j) {
  //     double expected = j + 1;
  //     ASSERT_NEAR(coefficients[j], expected, TOLERANCE)
  //         << "Coefficient mismatch at index " << j;
  //   }
  // }
}

// テストケースのパラメータを設定
INSTANTIATE_TEST_SUITE_P(
    SVMRegressionTests, SVMRegressionTest,
    ::testing::Combine(::testing::Values(100, 250, 500, 1000, 5000, 10000), // M: サンプル数
                       ::testing::Values(1, 2, 5, 10), // N: 特徴量数
                       ::testing::Values(0.1, 1.0, 10.0),   // C
                       ::testing::Values(0.01, 0.1, 1),         // ε（epsilon）
                       ::testing::Values(KernelType::LINEAR,
                                         KernelType::POLYNOMIAL,
                                         KernelType::RBF,
                                         KernelType::APPROX_RBF), // カーネル種
                       ::testing::Values(0.1),                    // gamma
                       ::testing::Values(3),                      // degree
                       ::testing::Values(1.0),                    // coef0
                       ::testing::Values(100)                     // approx_dim
                       ));
