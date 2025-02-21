#ifndef NEWTON_CG_SOLVER_HPP
#define NEWTON_CG_SOLVER_HPP

#include <vector>
#include <cstddef>
#include <functional>

/**
 * @class NewtonCGSolver
 * @brief
 * \if Japanese
 * ニュートン共役勾配法 (Newton-CG) ソルバー
 *
 * ヘッセ行列の明示的な計算を行わず、ヘッセ行列とベクトルの積 (Hessian-vector product, HVP) を
 * 用いて準ニュートン法による最適化を行う。
 * \else
 * Newton Conjugate Gradient (Newton-CG) Solver
 *
 * Performs quasi-Newton optimization without explicitly computing the Hessian matrix,
 * instead using Hessian-vector products (HVP).
 * \endif
 */
class NewtonCGSolver
{
public:
    /**
     * @struct Options
     * @brief
     * \if Japanese
     * Newton-CG ソルバーの設定オプション
     * \else
     * Configuration options for the Newton-CG solver
     * \endif
     */
    struct Options
    {
        int max_iterations = 100;         ///< \if Japanese Newton 反復の最大回数 \else Maximum number of Newton iterations \endif
        double tolerance = 1e-6;          ///< \if Japanese 勾配ノルム収束条件 \else Convergence condition for gradient norm \endif
        int cg_max_iterations = 50;       ///< \if Japanese 共役勾配法 (CG) の最大反復回数 \else Maximum iterations for Conjugate Gradient (CG) \endif
        double cg_tolerance = 1e-4;       ///< \if Japanese CG の絶対収束条件（下限） \else Absolute convergence tolerance for CG (lower bound) \endif
        double cg_tolerance_factor = 0.1; ///< \if Japanese CG の相対収束係数： tol_cg = factor * ||grad|| \else Relative convergence factor for CG: tol_cg = factor * ||grad|| \endif
        double line_search_alpha = 1.0;   ///< \if Japanese 初期ステップ長 \else Initial step size \endif
        double line_search_rho = 0.5;     ///< \if Japanese ラインサーチの減衰率 \else Line search decay rate \endif
        double line_search_c = 1e-4;      ///< \if Japanese Armijo 条件定数 \else Armijo condition constant \endif
        bool use_preconditioning = false; ///< \if Japanese 前処理付き CG を使用する場合 true \else Enable preconditioned CG if true \endif
    };

    /**
     * @brief
     * \if Japanese
     * Newton-CG ソルバーのコンストラクタ
     * \else
     * Constructor for Newton-CG solver
     * \endif
     *
     * @param opts
     * \if Japanese Newton-CG のオプション設定 \else Configuration options for Newton-CG \endif
     */
    explicit NewtonCGSolver(const Options &opts);

    /**
     * @brief
     * \if Japanese
     * マトリックスフリー Newton-CG ソルバーを用いた最適化
     *
     * Hessian-vector 積 (HVP) を用いて、ヘッセ行列を明示的に計算せずに最適化を行う。
     * \else
     * Optimization using matrix-free Newton-CG solver
     *
     * Uses Hessian-vector products (HVP) to optimize without explicitly computing the Hessian matrix.
     * \endif
     *
     * @param theta
     * \if Japanese 初期推定値（最適解に更新される） \else Initial estimate (updated to optimal solution) \endif
     *
     * @param func
     * \if Japanese 目的関数（関数値と勾配を返す関数オブジェクト） \else Objective function (returns function value and gradient) \endif
     *
     * @param hvp_func
     * \if Japanese Hessian-vector 積の計算関数（Hessian-vector product, HVP） \else Function for computing Hessian-vector products (HVP) \endif
     *
     * @return
     * \if Japanese 最適化後の目的関数値 \else Final objective function value after optimization \endif
     */
    double solve(std::vector<double> &theta,
                 std::function<double(const std::vector<double> &, std::vector<double> &)> func,
                 std::function<void(const std::vector<double> &, const std::vector<double> &, std::vector<double> &)> hvp_func);

private:
    Options options_; ///< \if Japanese Newton-CG のオプション設定 \else Configuration options for Newton-CG \endif

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
     * マトリックスフリー版 共役勾配法 (CG) によるヘッセ行列の近似解法
     * \else
     * Matrix-free Conjugate Gradient (CG) method for Hessian approximation
     * \endif
     *
     * @param theta
     * \if Japanese 最適化パラメータ \else Optimization parameters \endif
     *
     * @param b
     * \if Japanese CG の右辺（通常は -grad） \else Right-hand side for CG (typically -grad) \endif
     *
     * @param n
     * \if Japanese 変数の次元数 \else Dimension of variables \endif
     *
     * @param hvp_func
     * \if Japanese Hessian-vector 積を計算する関数 \else Function for computing Hessian-vector products \endif
     *
     * @param tol_cg
     * \if Japanese CG の収束許容誤差 \else Convergence tolerance for CG \endif
     *
     * @return
     * \if Japanese 解ベクトル \else Solution vector \endif
     */
    std::vector<double> conjugate_gradient_mf(
        const std::vector<double> &theta,
        const std::vector<double> &b,
        int n,
        std::function<void(const std::vector<double> &, const std::vector<double> &, std::vector<double> &)> hvp_func,
        double tol_cg);

    /**
     * @brief
     * \if Japanese
     * Armijo 条件を用いたバックトラックラインサーチ
     * \else
     * Backtracking line search using Armijo condition
     * \endif
     */
    double line_search(const std::vector<double> &theta,
                       const std::vector<double> &grad,
                       const std::vector<double> &d,
                       std::function<double(const std::vector<double> &, std::vector<double> &)> func,
                       double f,
                       double &new_f,
                       std::vector<double> &new_grad);
};

#endif // NEWTON_CG_SOLVER_HPP
