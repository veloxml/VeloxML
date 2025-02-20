#include <gtest/gtest.h>
#include "test_data_utils.hpp" // (X, Y) を生成する関数 generate_data() を含む
#include <random>
#include <vector>
#include <cmath>
#include "linear/ridge_regression.hpp" // RidgeRegression クラスのヘッダ

// 許容誤差（double の精度を考慮）
constexpr double TOLERANCE = 1e-2;

// Ridge回帰のテストクラス
// パラメータは、(サンプル数 M, 特徴量数 N, 正則化パラメータ lambda, penalize_bias フラグ)
class RidgeRegressionTest : public ::testing::TestWithParam<std::tuple<int, int, double, bool>>
{
};

TEST_P(RidgeRegressionTest, FitPredictTest)
{
    // パラメータ取得
    int M = std::get<0>(GetParam());
    int N = std::get<1>(GetParam());
    double lambda = std::get<2>(GetParam());
    bool penalize_bias = std::get<3>(GetParam());

    // サンプル数が特徴量数より小さい場合はテストをスキップ
    if (M < N)
    {
        GTEST_SKIP() << "Skipping test: M (" << M << ") < N (" << N << ")";
    }

    // データ生成
    // generate_data(M, N) は (M x N) の Col-Major 配列 X と (M x 1) の配列 Y を返すとする
    auto [X, Y] = generate_data(M, N);

    // Ridge回帰モデルの生成（lambda が非常に小さい値なら、ほぼ線形回帰と同じ挙動が期待される）
    RidgeRegression model(lambda, penalize_bias);

    // 学習を実行：X は (M x N) の1次元 Col-Major 配列、Y は (M x 1) の配列
    ASSERT_NO_THROW(model.fit(X, Y, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));

    // 予測値を取得
    std::vector<double> Y_pred;
    ASSERT_NO_THROW(Y_pred = model.predict(X, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));

    // 予測値のサイズチェック（(M x 1) の配列となるはず）
    ASSERT_EQ(Y_pred.size(), static_cast<size_t>(M));

    // 予測値が NaN や Inf でないかチェック
    for (int i = 0; i < M; ++i)
    {
        ASSERT_TRUE(std::isfinite(Y_pred[i])) << "Prediction at index " << i << " is non-finite";
    }

    // 予測誤差（MSE）の評価
    double mse = 0.0;
    for (int i = 0; i < M; ++i)
    {
        double diff = Y_pred[i] - Y[i];
        mse += diff * diff;
    }
    mse /= M;
    ASSERT_LT(mse, TOLERANCE) << "Mean Squared Error too high: " << mse;

    // 真の回帰係数（各特徴量の係数は j+1 として generate_data() で生成していると仮定）と推定係数の比較
    std::vector<double> coefficients = model.get_weights();
    ASSERT_EQ(coefficients.size(), static_cast<size_t>(N));
    for (int j = 0; j < N; ++j)
    {
        double expected = j + 1;
        ASSERT_NEAR(coefficients[j], expected, TOLERANCE) << "Coefficient mismatch at index " << j;
    }
}

// テストケースのパラメータを設定
INSTANTIATE_TEST_SUITE_P(
    RidgeRegressionTests,
    RidgeRegressionTest,
    ::testing::Combine(
        ::testing::Values(1000, 10000, 25000, 50000, 100000, 1000000), // M: サンプル数
        ::testing::Values(1, 2, 5, 10, 20, 50, 100),                   // N: 特徴量数
        ::testing::Values(1e-6, 1e-3),                                 // lambda 値（小さな値でほぼ線形回帰に近い）
        ::testing::Values(false, true)                                 // penalize_bias フラグ
        ));
