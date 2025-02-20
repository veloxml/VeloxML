#include <gtest/gtest.h>
#include "test_data_utils.hpp" // generate_data() などのテストデータ生成関数を定義していると仮定
#include <random>
#include <vector>
#include <cmath>
#include "linear/linear_regression.hpp" // 再実装した LinearRegression クラス

// 許容誤差（double の精度を考慮）
constexpr double TOLERANCE = 1e-2;

// 学習器のテストクラス
class LinearRegressionTest : public ::testing::TestWithParam<std::tuple<int, int, LinearDecompositionMode>>
{
};

TEST_P(LinearRegressionTest, FitPredictTest)
{
    // M: サンプル数, N: 特徴量数
    int M = std::get<0>(GetParam());
    int N = std::get<1>(GetParam());
    LinearDecompositionMode mode = std::get<2>(GetParam());

    if (M < N)
    {
        GTEST_SKIP() << "Skipping test due to insufficient samples M=" << M << " < N=" << N;
    }

    // generate_data() は (M×N) の Col‐Major 配列 X と (M×1) の配列 Y を返すものとする
    auto [X, Y] = generate_data(M, N);
    // ※ 必要に応じて標準化なども実施可能（ここではコメントアウト）

    // モデルの作成
    LinearRegression model(mode);

    // 学習実行：X は (M×N), Y は (M×1) の配列として渡す
    ASSERT_NO_THROW(model.fit(X, Y, static_cast<size_t>(M), static_cast<size_t>(N)));

    // 予測を取得
    std::vector<double> Y_pred;
    ASSERT_NO_THROW(Y_pred = model.predict(X, static_cast<size_t>(M), static_cast<size_t>(N)));

    // 予測結果のサイズチェック（(M×1) の配列）
    ASSERT_EQ(Y_pred.size(), static_cast<size_t>(M));

    // 予測値が NaN や Inf でないかチェック
    for (int i = 0; i < M; ++i)
    {
        ASSERT_TRUE(std::isfinite(Y_pred[i])) << "Prediction contains non-finite value at index " << i;
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

    // 真の回帰係数と推定値の比較
    std::vector<double> coefficients = model.get_weights();
    ASSERT_EQ(coefficients.size(), static_cast<size_t>(N));

    for (int j = 0; j < N; ++j)
    {
        double expected = j + 1; // generate_data() により各係数は j+1 として生成していると仮定
        ASSERT_NEAR(coefficients[j], expected, TOLERANCE) << "Coefficient mismatch at index " << j;
    }
}

// テストケースのパラメータを設定
INSTANTIATE_TEST_SUITE_P(
    LinearRegressionTests,
    LinearRegressionTest,
    ::testing::Combine(
        ::testing::Values(1000, 10000, 25000, 50000, 100000, 1000000),                                         // M: サンプル数
        ::testing::Values(1, 2, 5, 10, 20, 50, 100),                                                              // N: 特徴量数
        ::testing::Values(LinearDecompositionMode::LU, LinearDecompositionMode::QR, LinearDecompositionMode::SVD) // 分解モード
        // ::testing::Values(LinearDecompositionMode::LU) // 分解モード
    )
);
