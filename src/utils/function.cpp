#include <vector>
#include <cmath>
#include <iostream>

// シグモイド関数の実装
double sigmoid(double z)
{
  if (z >= 0)
  {
    double ez = std::exp(-z);
    return 1.0 / (1.0 + ez);
  }
  else
  {
    double ez = std::exp(z);
    return ez / (1.0 + ez);
  }
}

// ログ損失計算関数
double compute_logloss(std::vector<std::vector<double>> &Y_true, std::vector<std::vector<double>> &Y_pred)
{
  double loss = 0.0;
  int M = Y_true.size();

  for (int i = 0; i < M; ++i)
  {
    double p = std::max(std::min(Y_pred[i][0], 1.0 - 1e-15), 1e-15); // 確率が0/1にならないようクリップ
    loss += Y_true[i][0] * std::log(p) + (1 - Y_true[i][0]) * std::log(1 - p);
  }
  return -loss / M;
}

// Accuracy を計算する関数
double accuracy_score(std::vector<std::vector<double>>& y_true, std::vector<std::vector<double>>& y_pred) {
  if (y_true.size() != y_pred.size() || y_true.empty()) {
      throw std::invalid_argument("サイズが不正です。");
  }
  
  int correct = 0;
  for (size_t i = 0; i < y_true.size(); ++i) {
      if (y_true[i] == y_pred[i]) {
          ++correct;
      }
  }
  return static_cast<double>(correct) / y_true.size();
}

// F1 スコアを計算する関数（正例: 1, 負例: 0 と仮定）
double f1_score(std::vector<std::vector<double>>& y_true, std::vector<std::vector<double>>& y_pred) {
  if (y_true.size() != y_pred.size() || y_true.empty()) {
      throw std::invalid_argument("サイズが不正です。");
  }
  
  int true_positive = 0;
  int false_positive = 0;
  int false_negative = 0;
  
  for (size_t i = 0; i < y_true.size(); ++i) {
      if (y_pred[i][0] == 1 && y_true[i][0] == 1) {
          ++true_positive;
      } else if (y_pred[i][0] == 1 && y_true[i][0] == 0) {
          ++false_positive;
      } else if (y_pred[i][0] == 0 && y_true[i][0] == 1) {
          ++false_negative;
      }
  }
  
  double precision = (true_positive + false_positive) == 0 ?
                       0.0 : static_cast<double>(true_positive) / (true_positive + false_positive);
  double recall = (true_positive + false_negative) == 0 ?
                    0.0 : static_cast<double>(true_positive) / (true_positive + false_negative);
  
  if (precision + recall == 0.0) {
      return 0.0;
  }
  
  return 2 * precision * recall / (precision + recall);
}