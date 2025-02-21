#include <gtest/gtest.h>
#include "test_data_utils.hpp" // 既存のデータ生成ユーティリティ（X を生成）
#include <random>
#include <vector>
#include <cmath>
#include "pca/pca.hpp" // PCAクラスのインクルード

// 許容誤差（doubleの精度を考慮）
constexpr double TOLERANCE = 1e-5;

// PCAテストのパラメータ
class PCATest : public ::testing::TestWithParam<std::tuple<int, int, int>>
{
};

TEST_P(PCATest, FitTransformPredictTest)
{
  int M = std::get<0>(GetParam());
  int N = std::get<1>(GetParam());
  int n_components = std::get<2>(GetParam());

  if (n_components > N)
  {
    GTEST_SKIP() << "Skipping test: n_components (" << n_components << ") > N (" << N << ")";
  }

  // データ生成
  auto [X, Y] = generate_data(M, N);

  // PCA モデルの作成
  PCA model(n_components);

  // 学習（例外が発生しないこと）
  ASSERT_NO_THROW(model.fit(X, M, N));

  // 平均ベクトルのサイズチェック
  const std::vector<double> &mean = model.get_mean();
  ASSERT_EQ(mean.size(), static_cast<size_t>(N));

  // 主成分行列のサイズチェック (Col-Major: N × n_components)
  const std::vector<double> &components = model.get_components();
  ASSERT_EQ(components.size(), static_cast<size_t>(N * n_components));

  // transform のテスト：入力 X を主成分空間に射影
  std::vector<double> X_transformed;
  ASSERT_NO_THROW(X_transformed = model.transform(X, M, N));

  // 出力サイズのチェック (M × n_components)
  ASSERT_EQ(X_transformed.size(), static_cast<size_t>(M * n_components));

  // transform の出力に NaN や Inf が含まれないかを確認
  for (size_t i = 0; i < X_transformed.size(); ++i)
  {
    ASSERT_TRUE(std::isfinite(X_transformed[i]))
        << "Non-finite value in transformed data at index " << i;
  }

  // predict のテスト：predict は transform の出力と同一であるはず
  std::vector<double> X_pred;
  ASSERT_NO_THROW(X_pred = model.predict(X, M, N));
  ASSERT_EQ(X_pred.size(), X_transformed.size());

  for (size_t i = 0; i < X_pred.size(); ++i)
  {
    ASSERT_NEAR(X_pred[i], X_transformed[i], TOLERANCE)
        << "Mismatch between predict and transform at index " << i;
  }
}

// テストケースのパラメータを設定
INSTANTIATE_TEST_SUITE_P(
    PCATests,
    PCATest,
    ::testing::Combine(
        ::testing::Values(1000, 10000, 50000, 100000, 100000), // M: サンプル数
        ::testing::Values(10, 25, 50, 100),          // N: 特徴量数
        ::testing::Values(1, 2, 3, 5)          // n_components: 抽出する主成分数
        ));
