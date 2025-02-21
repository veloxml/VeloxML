#include "solver/newton_cg.hpp"
#include <cmath>
#include <iostream>
#include <cblas.h>
#include <omp.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

// コンストラクタ
NewtonCGSolver::NewtonCGSolver(const Options &opts)
    : options_(opts)
{
}

// 基本的な内積計算
double NewtonCGSolver::dot_product(const std::vector<double> &a, const std::vector<double> &b) const
{
    int n = static_cast<int>(a.size());
    return cblas_ddot(n, a.data(), 1, b.data(), 1);
}

// axpy: y += alpha * x
void NewtonCGSolver::axpy(double alpha, const std::vector<double> &x, std::vector<double> &y) const
{
    int n = static_cast<int>(x.size());
    cblas_daxpy(n, alpha, x.data(), 1, y.data(), 1);
}

// スケーリング: v *= alpha
void NewtonCGSolver::scale_vector(std::vector<double> &v, double alpha) const
{
    int n = static_cast<int>(v.size());
    cblas_dscal(n, alpha, v.data(), 1);
}

// ノルム計算
double NewtonCGSolver::norm(const std::vector<double> &v) const
{
    int n = static_cast<int>(v.size());
    return cblas_dnrm2(n, v.data(), 1);
}

// ----- マトリックスフリー版 CG 解法 -----
// 内部 CG で、H(theta)*d = b を、HVP 関数のみで解く。
// tol_cg は相対収束条件（例えば、options_.cg_tolerance_factor * ||b||）。
std::vector<double> NewtonCGSolver::conjugate_gradient_mf(
    const std::vector<double> &theta,
    const std::vector<double> &b,
    int n,
    std::function<void(const std::vector<double> &, const std::vector<double> &, std::vector<double> &)> hvp_func,
    double tol_cg)
{
    std::vector<double> x(n, 0.0); // 初期解 0
    std::vector<double> r = b;     // r = b - H*x (x=0)
    std::vector<double> p = r;     // p 初期値 = r
    double rsold = dot_product(r, r);

    // （オプション）前条件付き CG：Jacobi 前条件
    std::vector<double> M(n, 1.0); // 前条件（デフォルトは単位行列）
    if (options_.use_preconditioning)
    {
        // 簡易な Jacobi 前条件: M_i = 1 / (|H_{ii}| + eps)
        const double eps = 1e-8;
        for (int i = 0; i < n; i++)
        {
            // 単位ベクトル e_i
            std::vector<double> e(n, 0.0);
            e[i] = 1.0;
            std::vector<double> He(n, 0.0);
            hvp_func(theta, e, He);
            double diag = std::fabs(He[i]);
            M[i] = 1.0 / (diag + eps);
        }
        // 前条件適用: p = M .* r
        for (int i = 0; i < n; i++)
        {
            p[i] = M[i] * r[i];
        }
        rsold = 0.0;
        for (int i = 0; i < n; i++)
        {
            rsold += r[i] * p[i];
        }
    }

    for (int i = 0; i < options_.cg_max_iterations; i++)
    {
        std::vector<double> Ap(n, 0.0);
        // Ap = H * p を hvp_func で計算
        hvp_func(theta, p, Ap);

        double alpha = rsold / dot_product(p, Ap);
        // x = x + alpha * p
        axpy(alpha, p, x);
        // r = r - alpha * Ap
        axpy(-alpha, Ap, r);

        double rnorm = norm(r);
        if (rnorm < tol_cg)
            break;

        double rsnew;
        if (options_.use_preconditioning)
        {
            // r_tilde = M .* r
            std::vector<double> r_tilde(n, 0.0);
            for (int j = 0; j < n; j++)
            {
                r_tilde[j] = M[j] * r[j];
            }
            rsnew = 0.0;
            for (int j = 0; j < n; j++)
            {
                rsnew += r[j] * r_tilde[j];
            }
            double beta = rsnew / rsold;
            // p = r_tilde + beta * p
            for (int j = 0; j < n; j++)
            {
                p[j] = r_tilde[j] + beta * p[j];
            }
            rsold = rsnew;
        }
        else
        {
            rsnew = dot_product(r, r);
            if (rsnew < tol_cg * tol_cg)
                break;
            double beta = rsnew / rsold;
            // p = r + beta * p
            for (int j = 0; j < n; j++)
            {
                p[j] = r[j] + beta * p[j];
            }
            rsold = rsnew;
        }
    }
    return x;
}

// ----- ラインサーチ -----
// theta_candidate = theta + alpha*d を BLAS で高速に計算し、Armijo 条件を満たすステップサイズ alpha を探す。
double NewtonCGSolver::line_search(
    const std::vector<double> &theta,
    const std::vector<double> &grad,
    const std::vector<double> &d,
    std::function<double(const std::vector<double> &, std::vector<double> &)> func,
    double f,
    double &new_f,
    std::vector<double> &new_grad)
{
    const double c = options_.line_search_c;
    const double rho = options_.line_search_rho;
    const double alpha_init = options_.line_search_alpha;
    const int max_ls_iter = 50;
    double alpha = alpha_init;

    // BLAS で内積計算： grad^T * d
    double grad_dot_d = cblas_ddot(static_cast<int>(grad.size()), grad.data(), 1, d.data(), 1);

    std::vector<double> theta_candidate(theta); // 作業用領域
    int ls_iter = 0;
    while (ls_iter < max_ls_iter)
    {
        // theta_candidate = theta + alpha*d
        theta_candidate = theta;
        cblas_daxpy(static_cast<int>(theta.size()), alpha, d.data(), 1, theta_candidate.data(), 1);

        new_f = func(theta_candidate, new_grad);
        if (new_f <= f + c * alpha * grad_dot_d)
            break;
        alpha *= rho;
        ls_iter++;
    }
    if (ls_iter == max_ls_iter)
    {
        std::cerr << "Warning: Line search did not converge after " << max_ls_iter << " iterations." << std::endl;
    }
    return alpha;
}

// ----- Newton-CG solve -----
// この solve 関数では、Hessian の全体を構築せず、必要な Hessian-vector 積を hvp_func を介して計算するマトリックスフリー実装を行う。
double NewtonCGSolver::solve(std::vector<double> &theta,
                             std::function<double(const std::vector<double> &, std::vector<double> &)> func,
                             std::function<void(const std::vector<double> &, const std::vector<double> &, std::vector<double> &)> hvp_func)
{
    const int n = static_cast<int>(theta.size());
    std::vector<double> grad(n, 0.0);
    double f = func(theta, grad);
    double grad_norm = norm(grad);
    int iter = 0;

    // 作業用配列の再利用
    std::vector<double> new_grad(n, 0.0);
    std::vector<double> d(n, 0.0);
    std::vector<double> minus_grad(n, 0.0);

    while (grad_norm > options_.tolerance && iter < options_.max_iterations)
    {
        // 動的 CG 終了条件: tol_cg = cg_tolerance_factor * ||grad||
        double tol_cg = options_.cg_tolerance_factor * grad_norm;
        if (tol_cg < options_.cg_tolerance)
        {
            tol_cg = options_.cg_tolerance;
        }

        // 右辺 b = -grad
        minus_grad = grad;
        scale_vector(minus_grad, -1.0);

        // マトリックスフリー CG により、H * d = -grad を解く
        d = conjugate_gradient_mf(theta, minus_grad, n, hvp_func, tol_cg);

        // 降下方向チェック： d^T * grad 必ず負でなければならない
        if (dot_product(d, grad) >= 0)
        {
            d = grad;
            scale_vector(d, -1.0);
        }

        // ラインサーチ
        double new_f;
        double alpha = line_search(theta, grad, d, func, f, new_f, new_grad);

        // 更新: theta = theta + alpha * d
        axpy(alpha, d, theta);

        // 更新後の値を反映
        f = new_f;
        grad = new_grad;
        grad_norm = norm(grad);
        iter++;
        // std::cout << "Newton-CG Iteration " << iter
        //           << " f=" << f
        //           << " ||grad||=" << grad_norm << std::endl;
    }
    return f;
}
