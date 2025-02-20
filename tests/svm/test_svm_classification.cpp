// svm_classifier_test.cpp

#include <gtest/gtest.h>
#include "test_data_utils.hpp" // ダミーデータ生成や標準化用のユーティリティ
#include "utils/function.hpp"  // accuracy_score()、convertColMajorTo2D()などの関数
#include "utils/utilities.hpp"
#include "svm/svm_classification.hpp" // SVMClassificationのヘッダ
#include <tuple>
#include <vector>
#include <omp.h>
#include <iostream>

// 許容誤差（数値誤差を考慮）
constexpr double ACCURACY_TOLERANCE = 0.5;

// パラメータ化テスト： (M, N, kernel, approx_kernel)
//   - M : サンプル数
//   - N : 特徴数
//   - kernel : "linear", "rbf", "poly" のいずれか
//   - approx_kernel : 近似カーネル法を用いるかどうか
class SVMClassificationTest : public ::testing::TestWithParam<std::tuple<int, int, std::string, bool>>
{
};

TEST_P(SVMClassificationTest, FitPredictTest)
{
  // 遅いので、一旦テストスキップする
  GTEST_SKIP() << "Skipping test due to insufficient speed";

  int M = std::get<0>(GetParam());
  int N = std::get<1>(GetParam());
  std::string kernel = std::get<2>(GetParam());
  bool approx_kernel = std::get<3>(GetParam());

  // サンプル数が特徴数未満の場合はテストをスキップ
  if (M < N)
  {
    GTEST_SKIP() << "Skipping test due to insufficient samples: M=" << M << " < N=" << N;
  }

  if (approx_kernel && (kernel == "linear"))
  {
    GTEST_SKIP() << "Skipping test due to combine linear kernel and approx_kernel";
  }

  std::cout << "Kernel: " << kernel << ", approx_kernel: " << approx_kernel
            << ", M x N: " << M << " x " << N << std::endl;

  // ダミーデータ生成 (X: 2次元, Y: 1次元)
  // ※ generate_dummy_data_for_classificationは、(M x N)の行列Xと(M x 1)の行列Yを返すと仮定
  auto [X, Y] = generate_dummy_data_for_classification(M, N);
  standardize(X);

  // ColMajorな1次元配列に変換
  std::vector<double> X_flat = flattenMatrix(X);
  std::vector<double> Y_flat = flattenMatrix(Y); // Yは (M x 1) の行列と仮定

  // SVMClassificationではラベルを +1 / -1 として扱うため、
  // もしY_flatが {0, 1} であれば 0→ -1 に変換する
  for (auto &label : Y_flat)
  {
    if (label == 0)
      label = -1;
  }

  // モデル作成
  // コンストラクタ引数: (C, tol, max_passes, kernel, gamma, coef0, degree, approx_kernel)
  // ※ polyカーネルの場合、coef0やdegreeを設定します。ここではgamma=0.1, coef0=1.0, degree=3とします。
  SVMClassification svm(1.0, 1e-3, 100, kernel, true, 0.1, 1.0, 3, approx_kernel);

  // 学習の実行
  ASSERT_NO_THROW(svm.fit(X_flat, Y_flat, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));

  // 予測結果の取得
  std::vector<double> predictions;
  ASSERT_NO_THROW(predictions = svm.predict(X_flat, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));

  // 予測確率の取得
  std::vector<double> probabilities;
  ASSERT_NO_THROW(probabilities = svm.predict_proba(X_flat, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));

  for (int i = 0; i < M; i++)
  {
    if (predictions[i] == -1)
      predictions[i] = 0;
  }

  // 2次元配列に変換してaccuracy_score()を計算（予測結果は (M x 1) の形式）
  std::vector<std::vector<double>> pred2D = convertColMajorTo2D(predictions, static_cast<std::size_t>(M), 1);
  double acc = accuracy_score(Y, pred2D);

  ASSERT_GT(acc, ACCURACY_TOLERANCE) << "Accuracy too low: " << acc;
}

// テストケースのパラメータ設定
INSTANTIATE_TEST_SUITE_P(
    SVMTests,
    SVMClassificationTest,
    ::testing::Combine(
        ::testing::Values(1000, 2500),                                                     // M: サンプル数
        ::testing::Values(1, 5),                                                           // N: 特徴数
        ::testing::Values(std::string("linear"), std::string("rbf"), std::string("poly")), // カーネル種別
        ::testing::Values(false, true)                                                     // approx_kernel の有無
        ));
