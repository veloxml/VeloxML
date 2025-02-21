#ifndef TEST_DATA_UTILS_H
#define TEST_DATA_UTILS_H

#include <vector>
#include <utility>

void generate_data_for_trees(int num_samples, int num_features, std::vector<double>& X, std::vector<double>& Y);

// 回帰用ダミーデータ生成関数
std::pair<std::vector<double>, std::vector<double>> generate_data(int M, int N);
// 分類用ダミーデータ生成関数
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_dummy_data_for_classification(int M, int N);

void standardize(std::vector<std::vector<double>> &X);

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_cluster_data(int M, int N, int K);

std::vector<double> generate_cluster_data_flattened(int M, int N, int K);

#endif // TEST_DATA_UTILS_H
