// logistic_regression_test.cpp

#include <gtest/gtest.h>
#include "test_data_utils.hpp"    // ダミーデータ生成や標準化用のユーティリティ
#include "utils/function.hpp"      // compute_logloss()などの関数
#include "utils/utilities.hpp"
#include "linear/logistic_regression.hpp"
#include <tuple>
#include <vector>
#include <omp.h>
#include <iostream>

// 許容誤差 (数値誤差を考慮)
constexpr double ACCURACY_TOLERANCE = 0.5;

// パラメータ化テスト： (M, N, solver)
//   - M : サンプル数
//   - N : 特徴数
//   - solver : ソルバー種別 (LBFGS, NEWTON, CDは実装に合わせて)
class LogisticRegressionTest : public ::testing::TestWithParam<std::tuple<int, int, LogisticRegressionSolverType>> {
};

TEST_P(LogisticRegressionTest, FitPredictTest)
{
    int M = std::get<0>(GetParam());
    int N = std::get<1>(GetParam());
    LogisticRegressionSolverType solver = std::get<2>(GetParam());

    // サンプル数が特徴数未満の場合はテストをスキップ
    if (M < N) {
        GTEST_SKIP() << "Skipping test due to insufficient samples: M=" << M << " < N=" << N;
    }

    // ダミーデータ生成 (X: 2次元, Y: 1次元)
    auto [X, Y] = generate_dummy_data_for_classification(M, N);
    // 標準化 (行ごとに各特徴量のスケールを合わせる)
    standardize(X);

    // 内部処理用に、XをCol-Majorの1次元配列に変換
    std::vector<double> X_flat = flattenMatrix(X);
    std::vector<double> Y_flat = flattenMatrix(Y);  // Y は (n x 1) の行列と仮定
    // Yはすでに1次元配列であると仮定

    // モデル作成
    // コンストラクタ引数: (solver, lambda, tol, maxIter, ls_alpha_init, ls_rho, ls_c)
    LogisticRegression model(solver, 0.1, 1e-2, 10000, 1.0, 0.9, 1e-1);

    // 学習の実行
    // fit() の引数は (X_flat, Y, n, m) となる
    ASSERT_NO_THROW(model.fit(X_flat, Y_flat, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));


    // 予測結果の取得
    std::vector<double> predictions;
    ASSERT_NO_THROW(predictions = model.predict(X_flat, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));

    // 予測確率の取得
    std::vector<double> probabilities;
    ASSERT_NO_THROW(probabilities = model.predict_proba(X_flat, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));

    std::vector<std::vector<double>> pred2D = convertColMajorTo2D(predictions, static_cast<std::size_t>(M), 1);

    // Logloss のチェック（compute_loglossはラベル(Y)と確率(probabilities)を引数に取るものとする）
    double acc = accuracy_score(Y, pred2D);
    ASSERT_GT(acc, ACCURACY_TOLERANCE) << "Accuracy too low: " << acc;

    std::cout << acc << std::endl;
}

// テストケースのパラメータ設定
INSTANTIATE_TEST_SUITE_P(
    LogisticRegressionTests,
    LogisticRegressionTest,
    ::testing::Combine(
        ::testing::Values(1000, 10000, 25000, 50000, 100000, 1000000),                   // M: サンプル数
        ::testing::Values(1, 2, 5, 10, 20, 50),                                             // N: 特徴数
        ::testing::Values(LogisticRegressionSolverType::LBFGS, LogisticRegressionSolverType::NEWTON)
        // 必要に応じて、CDソルバーなども追加可能
    )
);
