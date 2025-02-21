#ifndef FUNCTION_UTILS_H
#define FUNCTION_UTILS_H

#include <vector>

double sigmoid(double z);
double compute_logloss(std::vector<std::vector<double>>& Y_true, std::vector<std::vector<double>>& Y_pred);
double accuracy_score(std::vector<std::vector<double>>& Y_true, std::vector<std::vector<double>>& Y_pred);;
double f1_score(std::vector<std::vector<double>>& Y_true, std::vector<std::vector<double>>& Y_pred);;

#endif // FUNCTION_UTILS_H