#include "function.hpp"
#include "test_data_utils.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

#include <cblas.h>
#include <omp.h>
#include <tbb/parallel_for.h>

// データ生成関数 (ColMajor, OpenMP + TBB 並列化)
void generate_data_for_trees(int num_samples, int num_features, std::vector<double> &X, std::vector<double> &Y)
{
    // 乱数生成器の設定
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    std::uniform_int_distribution<int> category_dist(0, 2);

    int num_numeric = num_features - 5; // 数値特徴量の数
    int num_categorical = 5;            // カテゴリカル特徴量の数

    X.resize(num_samples * num_features);
    Y.resize(num_samples);

    std::vector<double> X_numeric(num_samples * num_numeric);
    std::vector<int> X_categorical(num_samples * num_categorical);

// 数値特徴量の生成 (OpenMP 並列化)
#pragma omp parallel for
    for (int j = 0; j < num_numeric; ++j)
    {
        std::mt19937 local_gen(rd() + j);
        std::normal_distribution<double> local_dist(0.0, 1.0);
        for (int i = 0; i < num_samples; ++i)
        {
            X_numeric[j * num_samples + i] = local_dist(local_gen); // ColMajor 格納
        }
    }

// カテゴリカル特徴量の生成 (OpenMP 並列化)
#pragma omp parallel for
    for (int j = 0; j < num_categorical; ++j)
    {
        std::mt19937 local_gen(rd() + j);
        std::uniform_int_distribution<int> local_dist(0, 2);
        for (int i = 0; i < num_samples; ++i)
        {
            X_categorical[j * num_samples + i] = local_dist(local_gen); // ColMajor 格納
        }
    }

    // BLAS による行列積の計算 (X_numeric * W)
    std::vector<double> W(num_numeric, 1.0);
    std::vector<double> XW(num_samples, 0.0);

    cblas_dgemv(CblasColMajor, CblasNoTrans, num_samples, num_numeric, 1.0,
                X_numeric.data(), num_samples, W.data(), 1, 0.0, XW.data(), 1);

// X に 数値特徴 + カテゴリカル特徴を格納 (OpenMP 並列化)
#pragma omp parallel for
    for (int j = 0; j < num_numeric; ++j)
    {
        for (int i = 0; i < num_samples; ++i)
        {
            X[j * num_samples + i] = X_numeric[j * num_samples + i];
        }
    }

#pragma omp parallel for
    for (int j = 0; j < num_categorical; ++j)
    {
        for (int i = 0; i < num_samples; ++i)
        {
            X[(num_numeric + j) * num_samples + i] = static_cast<double>(X_categorical[j * num_samples + i]);
        }
    }

    // ターゲット Y の生成（TBB で並列化）
    tbb::parallel_for(0, num_samples, [&](int i)
                      {
        double x0 = X[0 * num_samples + i]; // X[:, 0] in ColMajor
        double x1 = X[1 * num_samples + i]; // X[:, 1] in ColMajor
        int cat_feature = static_cast<int>(X[(num_features - 1) * num_samples + i]); // 最後のカテゴリ特徴

        Y[i] = ((x0 * x0 + x1 * x1 > 1.5) && (cat_feature == 1)) ? 1.0 : 0.0; });
}

// X と Y の相関係数を計算する関数
std::vector<double> compute_correlation_with_target(const std::vector<double> &X, const std::vector<double> &Y, int M, int N)
{
    std::vector<double> correlation_vector(N, 0.0);

    // X の平均と分散を計算
    std::vector<double> mean_X(N, 0.0);
    std::vector<double> var_X(N, 0.0);
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            mean_X[j] += X[j * M + i];
        }
        mean_X[j] /= M;
    }
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            var_X[j] += (X[j * M + i] - mean_X[j]) * (X[j * M + i] - mean_X[j]);
        }
        var_X[j] /= (M - 1);
    }

    // Y の平均と分散を計算
    double mean_Y = 0.0;
    for (int i = 0; i < M; ++i)
    {
        mean_Y += Y[i];
    }
    mean_Y /= M;

    double var_Y = 0.0;
    for (int i = 0; i < M; ++i)
    {
        var_Y += (Y[i] - mean_Y) * (Y[i] - mean_Y);
    }
    var_Y /= (M - 1);

    // X と Y の相関係数を計算
    for (int j = 0; j < N; ++j)
    {
        double cov = 0.0;
        for (int i = 0; i < M; ++i)
        {
            cov += (X[j * M + i] - mean_X[j]) * (Y[i] - mean_Y);
        }
        cov /= (M - 1);

        correlation_vector[j] = cov / (std::sqrt(var_X[j]) * std::sqrt(var_Y));
    }

    return correlation_vector;
}

// 相関ベクトルを表示する
void print_correlation_with_target(const std::vector<double> &vector)
{
    std::cout << "\nFeature Correlation with Y:\n";
    for (size_t j = 0; j < vector.size(); ++j)
    {
        std::cout << "Feature " << (j + 1) << ": " << std::fixed << std::setprecision(3) << vector[j] << std::endl;
    }
}

std::pair<std::vector<double>, std::vector<double>>
generate_data(int M, int N)
{
    std::vector<double> X(M * N, 0.0);
    std::vector<double> Y(M, 0.0);

    // 乱数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal_dist(0.0, 0.01);
    std::uniform_real_distribution<double> uniform_dist(-5, 5);

    for (int i = 0; i < M; ++i)
    {
        double y_true = 0.0;
        for (int j = 0; j < N; ++j)
        {
            // Col-Major 順で要素を配置：列 j の各要素はインデックス j * M + i に格納される
            X[j * M + i] = uniform_dist(gen);
            y_true += (j + 1) * X[j * M + i]; // 真の回帰係数は j+1
        }
        // ノイズを加える場合は以下のように
        Y[i] = y_true + normal_dist(gen);
        // 必要に応じてノイズを加えない場合は、下記行を使用
        // Y[i] = y_true;
    }

    // std::vector<double> corr_vector = compute_correlation_with_target(X, Y, M, N);

    // // X と Y の相関を表示
    // print_correlation_with_target(corr_vector);

    return {X, Y};
}

// ロジスティック回帰用データ生成
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

// generate_cluster_data:
//  - M: サンプル数
//  - N: 特徴量数（次元数）
//  - K: クラスタ数
//
// 戻り値は、データ行列 X (M x N) と、各サンプルの真のクラスタラベル（0-indexed）のペア
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_cluster_data(int M, int N, int K)
{
    // 乱数エンジンの初期化
    std::random_device rd;
    std::mt19937 gen(rd());

    // 各クラスタの中心を決定
    // 各クラスタ中心は、全成分が (cluster_index+1)*100.0 とする
    std::vector<std::vector<double>> centers(K, std::vector<double>(N, 0.0));
    for (int k = 0; k < K; k++)
    {
        double center_value = (k + 1) * 100.0;
        for (int j = 0; j < N; j++)
        {
            centers[k][j] = center_value;
        }
    }

    // ノイズの標準偏差を設定（クラスタ内の分散を非常に小さくする）
    double noise_std = 0.001;
    std::normal_distribution<double> noise_dist(0.0, noise_std);

    // データ行列 X と真のクラスタラベルを初期化
    std::vector<std::vector<double>> X(M, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> true_labels(M, std::vector<double>(1, 0));

    // 各サンプルを生成
    // ここではラウンドロビン方式でクラスタ割り当てを行う（各クラスタに均等にサンプルを配置）
    for (int i = 0; i < M; i++)
    {
        int cluster_label = i % K;
        true_labels[i][0] = cluster_label;
        // 対応するクラスタ中心に、正規分布ノイズを加えてサンプルを生成
        for (int j = 0; j < N; j++)
        {
            X[i][j] = centers[cluster_label][j] + noise_dist(gen);
        }
    }
    return std::make_tuple(X, true_labels);
}

// データ生成関数 (Col-Major 形式の `std::vector<double>` を返す)
std::vector<double> generate_cluster_data_flattened(int M, int N, int K)
{
    // 各クラスタに均等にサンプルを割り当てる（M が K で割り切れると仮定）
    int samples_per_cluster = M / K;
    std::vector<double> X(M * N, 0.0);
    std::mt19937 gen(42);
    // クラスタごとのノイズの標準偏差
    std::normal_distribution<double> noise(0.0, 0.00001);

    // 各クラスタの中心を、例えば各特徴量で (cluster_index * offset) とする
    double offset = 10.0;
    std::vector<double> centers(K * N, 0.0);
    for (int k = 0; k < K; ++k)
    {
        for (int j = 0; j < N; ++j)
        {
            centers[k * N + j] = (k + 1) * offset; // クラスタ k の j 番目の特徴量
        }
    }

    // データ生成: 各クラスタごとに、中心にノイズを加える
    for (int k = 0; k < K; ++k)
    {
        for (int i = 0; i < samples_per_cluster; ++i)
        {
            int idx = k * samples_per_cluster + i;
            for (int j = 0; j < N; ++j)
            {
                // Col-Major 形式: 特徴量 j のサンプル idx は X[j * M + idx]
                X[j * M + idx] = centers[k * N + j] + noise(gen);
            }
        }
    }

    // 残りのサンプル（M が K で割り切れない場合）の処理（ここでは最初のクラスタに追加）
    for (int idx = samples_per_cluster * K; idx < M; ++idx)
    {
        for (int j = 0; j < N; ++j)
        {
            X[j * M + idx] = centers[0 * N + j] + noise(gen);
        }
    }
    return X;
}
