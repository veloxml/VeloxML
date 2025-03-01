#include <metal_stdlib>
using namespace metal;

// このカーネルは、各スレッドがミニバッチ内の1サンプル（1行）を処理する前提です。
kernel void lasso_sgd(
    device const float* X         [[ buffer(0) ]], // 入力データ（row-major: [batch_size x cols]）
    device const float* y         [[ buffer(1) ]], // 正解ラベル（サイズ：batch_size）
    device float* w               [[ buffer(2) ]], // 重み（サイズ：cols）
    constant float& lambda        [[ buffer(3) ]],
    constant float& lr            [[ buffer(4) ]],
    constant uint& batch_size     [[ buffer(5) ]],
    constant uint& cols           [[ buffer(6) ]],
    uint row_id                 [[ thread_position_in_grid ]])
{
    // スレッド数はバッチサイズと同じと仮定
    if (row_id >= batch_size) return;

    // 1サンプル分の予測計算（ドット積）
    float pred = 0.0;
    for (uint j = 0; j < cols; j++) {
        pred += X[row_id * cols + j] * w[j];
    }
    
    // 1サンプルの誤差
    float error = pred - y[row_id];
    
    // ミニバッチの平均をとるためのスケーリング係数（各サンプルの勾配は1/batch_sizeで重み更新）
    float scale = 1.0 / float(batch_size);
    
    // 各特徴量について勾配を計算し、重みを更新
    for (uint j = 0; j < cols; j++) {
        // サンプルiにおける勾配寄与: (error * x_ij) の平均にL1正則化項を加える
        float grad = error * X[row_id * cols + j] * scale;
        // L1正則化のサブ勾配（wがゼロの場合は0にする）
        float subgrad = (w[j] > 0.0 ? 1.0 : (w[j] < 0.0 ? -1.0 : 0.0)) * lambda;
        float update = lr * (grad + subgrad);
        // 複数スレッドからの更新が競合しないようにatomic操作を使用
        atomic_fetch_add_explicit((device atomic_float *)&w[j], -update, memory_order_relaxed);
    }
}
