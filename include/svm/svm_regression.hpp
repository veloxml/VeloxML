#ifndef SVM_REGRESSION_HPP
#define SVM_REGRESSION_HPP

#include "base/regression_base.hpp"
#include <stdexcept>
#include <vector>

// カーネル種の種類
enum class KernelType {
  LINEAR,
  POLYNOMIAL,
  RBF,
  APPROX_RBF // Random Fourier Features による近似 RBF
};

class SVMRegression : public RegressionBase{
public:
  // コンストラクタ:
  // C, epsilon, tol, max_iter: SMOのハイパーパラメータ
  // kernel_type: 使用するカーネル種
  // gamma, degree, coef0: POLYNOMIAL, RBF用のパラメータ
  // approx_dim: APPROX_RBF用の写像次元
  SVMRegression(double C, double epsilon, double tol, int max_iter,
                KernelType kernel_type, double gamma, int degree, double coef0,
                int approx_dim);
  ~SVMRegression();

  // 学習: Xは(n x m)のCol‐Major配列、Yは長さnの出力
  void fit(const std::vector<double> &X, const std::vector<double> &Y,
           std::size_t n, std::size_t m) override;

  // 予測: テストデータXは(n x m)のCol‐Major配列。内部で逆標準化して返す。
  std::vector<double> predict(const std::vector<double> &X, std::size_t n,
                              std::size_t m) override;

  // 補助関数
  std::vector<double> get_weights() const; // 線形/APPROX_RBF用のプライマル解
  double get_bias() const;
  std::vector<double> get_theta_dual() const; // 双対変数の差(α - α*)

private:
  // ハイパーパラメータ
  double C_;
  double epsilon_;
  double tol_;
  int max_iter_;
  KernelType kernel_type_;
  double gamma_;
  int degree_;
  double coef0_;
  int approx_dim_;

  // 学習済みパラメータ
  double bias_;
  std::vector<double> weights_; // LINEAR, APPROX_RBF用
  std::vector<double> theta_dual_; // POLYNOMIAL,RBF用（各学習サンプルのα-α*）

  // 学習時のデータ保存（非線形用）
  std::vector<double> X_train_; // Col‐Major, size: train_n_ x train_m_
  std::vector<double> Y_train_; // 標準化後の出力
  std::size_t train_n_, train_m_;

  // APPROX_RBF 用の Random Fourier Features のパラメータ
  std::vector<double> W_;        // size: approx_dim_ x m
  std::vector<double> rff_bias_; // size: approx_dim_

  // Yの標準化パラメータ（学習時に保存）
  double Y_mean_;
  double Y_std_;

  // 内部関数：カーネル計算（X_effはCol‐Major、m_effは有効次元）
  double kernel_function(const std::vector<double> &X_eff, std::size_t n,
                         std::size_t m_eff, std::size_t i, std::size_t j) const;

  // 内部関数：候補探索（TBBなどで実装可：ここでは省略）
  std::size_t selectSecondIndex(std::size_t i, const std::vector<double> &E,
                                const std::vector<double> &alpha,
                                const std::vector<double> &alpha_star) const;

  // 内部関数：更新ペアの処理（4通りのケースを網羅）
  bool updatePair(std::size_t i, std::size_t j,
                  const std::vector<double> &X_eff, std::size_t n,
                  std::size_t m_eff, std::vector<double> &alpha,
                  std::vector<double> &alpha_star, std::vector<double> &E);

  // 内部関数：全サンプルの双対ギャップ（KKT違反の最大値）を計算
  double computeDualGap(const std::vector<double> &X_eff, std::size_t n,
                        std::size_t m_eff, const std::vector<double> &alpha,
                        const std::vector<double> &alpha_star) const;
};

#endif // SVM_REGRESSION_HPP
