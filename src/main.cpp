#include "svm/svm_classification.hpp"
#include "utils/utilities.hpp"
#include "utils/function.hpp"

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
generate_dummy_data_for_classification(int M, int N)
{
    // 固定シードで再現性を確保
    std::mt19937 gen(42);
    // 特徴量は [-1, 1] の一様分布
    std::uniform_real_distribution<double> uniform_dist(-1, 1);
    // ノイズは小さめの標準偏差 (0.1)
    std::normal_distribution<double> normal_dist(0.0, 0.1);

    std::vector<std::vector<double>> X(M, std::vector<double>(N));
    std::vector<std::vector<double>> Y(M, std::vector<double>(1));

    for (int i = 0; i < M; ++i)
    {
        double linear_sum = 0.0;
        for (int j = 0; j < N; ++j)
        {
            X[i][j] = uniform_dist(gen);
            linear_sum += (j + 1) * X[i][j]; // 真の回帰係数は j+1
        }

        double prob = sigmoid(linear_sum + normal_dist(gen)); // ノイズ追加
        Y[i][0] = (prob >= 0.5) ? 1 : 0;                      // しきい値で分類
    }
    return {X, Y};
}

void standardize(std::vector<std::vector<double>> &X)
{
    int M = X.size(), N = X[0].size();
    std::vector<double> mean(N, 0.0), stddev(N, 0.0);

    // 平均値計算
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            mean[j] += X[i][j];
        }
        mean[j] /= M;
    }

    // 標準偏差計算
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            stddev[j] += (X[i][j] - mean[j]) * (X[i][j] - mean[j]);
        }
        stddev[j] = std::sqrt(stddev[j] / M);
        if (stddev[j] == 0.0)
            stddev[j] = 1.0; // ゼロ割防止
    }

    // 標準化
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            X[i][j] = (X[i][j] - mean[j]) / stddev[j];
        }
    }
}

int main()
{
    int M = 1000;
    int N = 5;
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
    SVMClassification svm(1.0, 1e-3, 100, "poly", true, 0.1, 1.0, 3, true);

    // 学習の実行
    svm.fit(X_flat, Y_flat, static_cast<std::size_t>(M), static_cast<std::size_t>(N));

    return 0;
}