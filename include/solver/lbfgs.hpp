#ifndef LBFGS_SOLVER_HPP
#define LBFGS_SOLVER_HPP

#include <vector>
#include <cstddef>
#include <functional>

/**
 * @class LBFGSSolver
 * @brief
 * \if Japanese
 * L-BFGS (Limited-memory BFGS) ソルバーの実装
 *
 * 目的関数 \( f(\theta) \) とその勾配を用いて最適化を行う。
 * 目的関数は以下のシグネチャを満たす必要があります：
 * ```cpp
 * double func(const std::vector<double>& theta, std::vector<double>& grad)
 * ```
 *
 * - `theta` は初期値として入力され、解に更新される。
 * - `grad` は `theta` における勾配を出力するためのベクトル。
 * - 戻り値は `f(theta)` の値を返す。
 * \else
 * Implementation of the L-BFGS (Limited-memory BFGS) solver
 *
 * Optimizes an objective function \( f(\theta) \) using its gradient.
 * The objective function must follow this signature:
 * ```cpp
 * double func(const std::vector<double>& theta, std::vector<double>& grad)
 * ```
 *
 * - `theta` is given as an initial value and updated as the solution.
 * - `grad` is an output vector storing the gradient at `theta`.
 * - The function returns the value of `f(theta)`.
 * \endif
 */
class LBFGSSolver
{
public:
    /**
     * @struct Options
     * @brief
     * \if Japanese
     * L-BFGSソルバーの設定オプション
     * \else
     * Configuration options for the L-BFGS solver
     * \endif
     */
    struct Options
    {
        int max_iterations = 100;       ///< \if Japanese 最大反復回数 \else Maximum number of iterations \endif
        double tolerance = 1e-6;        ///< \if Japanese 収束許容誤差 \else Convergence tolerance \endif
        int m = 10;                     ///< \if Japanese 履歴サイズ（メモリ制限付き） \else History size (limited-memory) \endif
        double line_search_alpha = 1.0; ///< \if Japanese 初期ステップ長 \else Initial step size \endif
        double line_search_rho = 0.5;   ///< \if Japanese ラインサーチの減衰率 \else Line search decay rate \endif
        double line_search_c = 1e-4;    ///< \if Japanese Armijo条件の定数 \else Armijo condition constant \endif
    };

    /**
     * @brief
     * \if Japanese
     * L-BFGSソルバーのコンストラクタ
     * \else
     * Constructor for L-BFGS solver
     * \endif
     *
     * @param opts
     * \if Japanese L-BFGS のオプション設定 \else Configuration options for L-BFGS \endif
     */
    explicit LBFGSSolver(const Options &opts);

    /**
     * @brief
     * \if Japanese
     * L-BFGS法による最適化を実行する
     * \else
     * Perform optimization using L-BFGS
     * \endif
     *
     * @param theta
     * \if Japanese 最適化変数（初期値として与えられ、解に更新される） \else Optimization variables (updated as the solution) \endif
     *
     * @param func
     * \if Japanese 目的関数（関数値と勾配を返す関数オブジェクト） \else Objective function (returns function value and gradient) \endif
     *
     * @return
     * \if Japanese 最適化後の目的関数値 \else Final objective function value after optimization \endif
     */
    double solve(std::vector<double> &theta,
                 std::function<double(const std::vector<double> &, std::vector<double> &)> func);

private:
    Options options_; ///< \if Japanese L-BFGS のオプション設定 \else Configuration options for L-BFGS \endif

    // L-BFGS の履歴情報
    std::vector<std::vector<double>> s_history_; ///< \if Japanese 過去の変数差分 \else History of variable differences \endif
    std::vector<std::vector<double>> y_history_; ///< \if Japanese 過去の勾配差分 \else History of gradient differences \endif
    std::vector<double> rho_history_;            ///< \if Japanese 1 / (y^T s) の履歴 \else History of 1 / (y^T s) values \endif

    /**
     * @brief
     * \if Japanese
     * ベクトルの内積を計算する
     * \else
     * Compute the dot product of two vectors
     * \endif
     */
    double dot_product(const std::vector<double> &a, const std::vector<double> &b) const;

    /**
     * @brief
     * \if Japanese
     * `y = y + alpha * x` を計算する（BLASのaxpyに相当）
     * \else
     * Compute `y = y + alpha * x` (equivalent to BLAS axpy)
     * \endif
     */
    void axpy(double alpha, const std::vector<double> &x, std::vector<double> &y) const;

    /**
     * @brief
     * \if Japanese
     * ベクトルのスカラー倍を計算する
     * \else
     * Scale a vector by a scalar
     * \endif
     */
    void scale_vector(std::vector<double> &v, double alpha) const;

    /**
     * @brief
     * \if Japanese
     * ベクトルのノルムを計算する
     * \else
     * Compute the norm of a vector
     * \endif
     */
    double norm(const std::vector<double> &v) const;

    /**
     * @brief
     * \if Japanese
     * Armijo条件に基づくバックトラックラインサーチを実行する
     * \else
     * Perform backtracking line search based on the Armijo condition
     * \endif
     *
     * @param theta
     * \if Japanese 現在の変数 \else Current variable values \endif
     *
     * @param grad
     * \if Japanese 現在の勾配 \else Current gradient \endif
     *
     * @param p
     * \if Japanese 探索方向ベクトル \else Search direction vector \endif
     *
     * @param func
     * \if Japanese 目的関数 \else Objective function \endif
     *
     * @param f
     * \if Japanese 現在の目的関数値 \else Current function value \endif
     *
     * @param new_f
     * \if Japanese 更新後の目的関数値（出力） \else Updated function value (output) \endif
     *
     * @param new_grad
     * \if Japanese 更新後の勾配（出力） \else Updated gradient (output) \endif
     *
     * @return
     * \if Japanese ラインサーチ後のステップサイズ \else Step size after line search \endif
     */
    double line_search(const std::vector<double> &theta,
                       const std::vector<double> &grad,
                       const std::vector<double> &p,
                       std::function<double(const std::vector<double> &, std::vector<double> &)> func,
                       double f,
                       double &new_f,
                       std::vector<double> &new_grad);
};

#endif // LBFGS_SOLVER_HPP
