#include <metal_stdlib>
using namespace metal;

// カーネル1: 各サンプルの予測値を計算する
kernel void compute_pred(
    device const float *X        [[ buffer(0) ]], // 特徴量行列 (rows x cols), row-major
    device const float *w        [[ buffer(1) ]], // 重みベクトル (cols)
    device float *pred           [[ buffer(2) ]], // 出力予測値 (rows)
    constant uint &rows          [[ buffer(3) ]],
    constant uint &cols          [[ buffer(4) ]],
    uint id                    [[ thread_position_in_grid ]]
) {
    if (id >= rows) return;
    float dot_val = 0.0;
    // vectorized version: 4要素ずつ処理（colsが4の倍数でない場合は後で調整）
    uint j = 0;
    for (; j + 3 < cols; j += 4) {
        float4 Xvec = *((const device float4*)(X + id * cols + j));
        float4 wvec = float4(w[j], w[j+1], w[j+2], w[j+3]);
        dot_val += dot(Xvec, wvec);
    }
    for (; j < cols; j++) {
        dot_val += X[id * cols + j] * w[j];
    }
    pred[id] = dot_val;
}

// カーネル2: 各係数ごとの勾配計算と soft-thresholding による更新
kernel void fista_lasso_update(
    device const float *X        [[ buffer(0) ]], // 特徴量行列 (rows x cols)
    device const float *y        [[ buffer(1) ]], // 目的変数 (rows)
    device const float *pred     [[ buffer(2) ]], // 予測値 (rows) ← compute_pred の結果
    device float *w              [[ buffer(3) ]], // 重み (cols)
    device float *w_prev         [[ buffer(4) ]], // 前回の重み (cols)
    device float *z              [[ buffer(5) ]], // FISTA用変数 (cols)
    constant float &lambda       [[ buffer(6) ]],
    constant float &lr           [[ buffer(7) ]],
    constant uint &rows          [[ buffer(8) ]],
    constant uint &cols          [[ buffer(9) ]],
    uint id                      [[ thread_position_in_grid ]]
) {
    if (id >= cols) return;
    float grad = 0.0;
    // 各サンプルについて (pred - y) * X[i, id] を積算
    for (uint i = 0; i < rows; i++) {
        grad += (pred[i] - y[i]) * X[i * cols + id];
    }
    grad /= rows;
    
    // soft-thresholding 付き更新
    float w_new = z[id] - lr * grad;
    w_new = sign(w_new) * max(0.0, fabs(w_new) - lambda * lr);
    
    // 更新用に w_prev に前回の w を保存
    w_prev[id] = w[id];
    // w の更新
    w[id] = w_new;
    // 暫定的に z も更新（ホスト側で加速更新を行う前の値）
    z[id] = w_new;
}
