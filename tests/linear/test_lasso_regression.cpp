#include <gtest/gtest.h>
#include "test_data_utils.hpp" // (X, Y) を生成する generate_data() 関数を含む
#include <random>
#include <vector>
#include <cmath>
#include "linear/lasso_regression.hpp" // LassoRegression クラスのヘッダ（パスは環境に合わせて調整してください）

// 許容誤差（doubleの精度を考慮）
constexpr double TOLERANCE = 1e-2;

// Lasso回帰のテストクラス
// パラメータ: サンプル数 M, 特徴量数 N, 正則化パラメータ lambda, ソルバーのモード
class LassoRegressionTest : public ::testing::TestWithParam<std::tuple<int, int, double, LassoSolverMode>>
{
};

TEST_P(LassoRegressionTest, FitPredictTest)
{
    int M = std::get<0>(GetParam());
    int N = std::get<1>(GetParam());
    double lambda = std::get<2>(GetParam());
    LassoSolverMode mode = std::get<3>(GetParam());

    // サンプル数が特徴量数より少ない場合はテストをスキップ
    if (M < N)
    {
        GTEST_SKIP() << "Skipping test: M (" << M << ") < N (" << N << ")";
    }

    // generate_data(M, N) は (M x N) の Col-Major 配列 X と (M x 1) の配列 Y を返す前提
    auto [X, Y] = generate_data(M, N);

    // LassoRegression モデルの生成
    // コンストラクタ: LassoRegression(lambda, max_iter, tol, mode, admm_rho, penalize_bias)
    // ここでは max_iter = 1000, tol = 1e-6, admm_rho = 1.0, penalize_bias = false と固定
    LassoRegression model(lambda, 1000, 1e-6, mode, 1.0, false);

    // 学習実行：X は (M x N) の 1次元 Col-Major 配列、Y は (M x 1) の配列
    ASSERT_NO_THROW(model.fit(X, Y, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));

    // 予測を取得
    std::vector<double> Y_pred;
    ASSERT_NO_THROW(Y_pred = model.predict(X, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));

    // 予測結果のサイズチェック（(M x 1) の配列）
    ASSERT_EQ(Y_pred.size(), static_cast<size_t>(M));

    // 予測値が NaN や Inf でないか確認
    for (int i = 0; i < M; ++i)
    {
        ASSERT_TRUE(std::isfinite(Y_pred[i])) << "Prediction at index " << i << " is non-finite";
    }

    // 予測誤差 (MSE) の評価
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
    LassoRegressionTests,
    LassoRegressionTest,
    ::testing::Combine(
        ::testing::Values(1000, 10000, 25000, 50000, 100000, 1000000),   // M: サンプル数
        ::testing::Values(1, 2, 5, 10, 20, 50, 100),                     // N: 特徴量数
        ::testing::Values(1e-6, 1e-4),                                   // lambda 値（小さい値ならほぼ線形回帰と同じ挙動が期待できる）
        ::testing::Values(LassoSolverMode::FISTA, LassoSolverMode::ADMM) // ソルバーのモード
        ));
