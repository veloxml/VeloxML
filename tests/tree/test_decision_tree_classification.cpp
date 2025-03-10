#include <gtest/gtest.h>
#include "test_data_utils.hpp"    // ダミーデータ生成や標準化用のユーティリティ
#include "utils/function.hpp"      // compute_logloss()などの関数
#include "utils/utilities.hpp"
#include "tree/decision_tree_classification.hpp"
#include <tuple>
#include <vector>
#include <omp.h>

// 許容誤差 (数値誤差を考慮)
constexpr double ACCURACY_TOLERANCE = 0.5;

// パラメータ化テスト： (M, N, solver)
//   - M : サンプル数
//   - N : 特徴数
class DecisionTreeClassificationTest : public ::testing::TestWithParam<std::tuple<int, int, Criterion, SplitAlgorithm>> {
};

TEST_P(DecisionTreeClassificationTest, FitPredictTest)
{
    int M = std::get<0>(GetParam());
    int N = std::get<1>(GetParam());
    Criterion criterion = std::get<2>(GetParam());
    SplitAlgorithm split_algorithm = std::get<3>(GetParam());

    // サンプル数が特徴数未満の場合はテストをスキップ
    if (M < N) {
        GTEST_SKIP() << "Skipping test due to insufficient samples: M=" << M << " < N=" << N;
    }

    // ダミーデータ生成 (X: 2次元, Y: 1次元)
    std::vector<double> X;
    std::vector<double> Y;
    generate_data_for_trees(M, N, X, Y);
    std::vector<std::vector<double>> Y_mat = convertColMajorTo2D(Y, static_cast<std::size_t>(M), 1);

    // モデル作成
    // コンストラクタ引数: (solver, lambda, tol, maxIter, ls_alpha_init, ls_rho, ls_c)
    DecisionTreeClassification model(5, 20, 256, criterion, split_algorithm, 2, 10, 1.0, 20);

    // 学習の実行
    // fit() の引数は (X_flat, Y, n, m) となる
    ASSERT_NO_THROW(model.fit(X, Y, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));

    // 予測結果の取得
    std::vector<double> predictions;
    ASSERT_NO_THROW(predictions = model.predict(X, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));
    // 予測確率の取得
    std::vector<double> probabilities;
    ASSERT_NO_THROW(probabilities = model.predict_proba(X, static_cast<std::size_t>(M), static_cast<std::size_t>(N)));
    std::vector<std::vector<double>> pred2D = convertColMajorTo2D(predictions, static_cast<std::size_t>(M), 1);

    // Logloss のチェック（compute_loglossはラベル(Y)と確率(probabilities)を引数に取るものとする）
    double acc = accuracy_score(Y_mat, pred2D);
    ASSERT_GT(acc, ACCURACY_TOLERANCE) << "Accuracy too low: " << acc;
}

// テストケースのパラメータ設定
INSTANTIATE_TEST_SUITE_P(
    DecisionTreeClassificationTests,
    DecisionTreeClassificationTest,
    ::testing::Combine(
        ::testing::Values(1000, 10000, 25000, 50000, 100000, 1000000),                   // M: サンプル数
        ::testing::Values(10, 20, 50, 100),                                             // N: 特徴数
        ::testing::Values(Criterion::Entropy, Criterion::Gini, Criterion::Logloss),
        ::testing::Values(SplitAlgorithm::Standard, SplitAlgorithm::Histogram)
    )
);
