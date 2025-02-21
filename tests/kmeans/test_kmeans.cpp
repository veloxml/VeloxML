#include <gtest/gtest.h>
#include "test_data_utils.hpp" // 例: generate_cluster_data_flattened
#include <vector>
#include <cmath>
#include <set>
#include "kmeans/kmeans.hpp"

// 許容誤差
constexpr double TOLERANCE = 1e-2;

// KMeans のパラメータ化テスト
class KMeansTest : public ::testing::TestWithParam<std::tuple<int, int, int, KMeansAlgorithm, bool>> {};

TEST_P(KMeansTest, FitPredictTransformTest)
{
    int M = std::get<0>(GetParam());
    int N = std::get<1>(GetParam());
    int K = std::get<2>(GetParam());
    KMeansAlgorithm algorithm = std::get<3>(GetParam());
    bool use_kdtree = std::get<4>(GetParam());

    // データ生成（Col-Major）
    std::vector<double> X = generate_cluster_data_flattened(M, N, K);

    // KMeans モデルの作成
    KMeans model(K, 300, 1e-4, algorithm, use_kdtree);

    // 学習（例外が発生しないこと）
    ASSERT_NO_THROW(model.fit(X, M, N));

    // 予測ラベル取得
    std::vector<double> predicted_labels;
    ASSERT_NO_THROW(predicted_labels = model.predict(X, M, N));
    ASSERT_EQ(predicted_labels.size(), static_cast<size_t>(M));

    // 返されるラベルは [0, K-1] の範囲であるべき
    for (int i = 0; i < M; ++i)
    {
        ASSERT_GE(predicted_labels[i], 0);
        ASSERT_LT(predicted_labels[i], K);
    }

    // クラスタ中心との距離取得
    std::vector<double> distances;
    ASSERT_NO_THROW(distances = model.transform(X, M, N));
    ASSERT_EQ(distances.size(), static_cast<size_t>(M * K));

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            ASSERT_TRUE(std::isfinite(distances[i * K + j]))
                << "Non-finite distance at (" << i << ", " << j << ")";
        }
    }

    // inertia（各サンプルの、割り当てられたクラスタ中心との二乗距離の平均）を計算
    double inertia = 0.0;
    for (int i = 0; i < M; ++i)
    {
        double d = distances[i * K + static_cast<int>(predicted_labels[i])];
        inertia += d * d;
    }
    inertia /= M;

    // inertia の値が閾値未満であることを確認
    ASSERT_LT(inertia, TOLERANCE) << "Mean Squared Distance (inertia) too high: " << inertia;

    // クラスタの数が K であることを確認
    std::set<int> unique_labels(predicted_labels.begin(), predicted_labels.end());
    ASSERT_EQ(unique_labels.size(), K);
}

// テストケースのパラメータ
INSTANTIATE_TEST_SUITE_P(
    KMeansTests,
    KMeansTest,
    ::testing::Combine(
        ::testing::Values(500, 1000, 5000, 10000, 25000, 50000), // サンプル数 M
        ::testing::Values(2, 5, 10, 25, 50, 100),               // 特徴量数 N
        ::testing::Values(2, 3, 5, 10, 25, 50),                 // クラスタ数 K
        ::testing::Values(KMeansAlgorithm::STANDARD, KMeansAlgorithm::ELKAN, KMeansAlgorithm::HAMERLY), // アルゴリズム
        ::testing::Values(false, true)                          // KDTree の使用有無
    ));
