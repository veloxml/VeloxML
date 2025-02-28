#include <metal_stdlib>
using namespace metal;

#define THREADGROUP_SIZE 256
#define EPSILON 1e-6

// このカーネルでは、拡大行列 A は device メモリ上に配置されているものとします。
// A のサイズは n_features x (n_features+1) で、左側 n_features×n_features が (X^T X) 部分（後に正則化λを加える）、
// 右端の列が X^T y を格納しています。
kernel void ridge_regression_large(device float*       A           [[ buffer(0) ]],
                                   constant uint &     n_features  [[ buffer(1) ]],
                                   constant float &    lambda      [[ buffer(2) ]],
                                   device uint*        error_flag  [[ buffer(3) ]],
                                   uint tid                   [[ thread_index_in_threadgroup ]]) {
    // --- LU分解（正則化＋部分ピボット付き消去） ---
    // 外側ループ：0～n_features-1 の各ピボット反復を逐次処理
    for (uint i = 0; i < n_features; i++) {
        // ① 対角成分に正則化 λ を加算（各反復の先頭で適用）
        if (tid == 0) {
            A[i * (n_features + 1) + i] += lambda;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // ② 部分ピボット選択
        // 各スレッドは、列 i の i 行目以降の中から自分の担当領域の最大値候補を探索し、リダクションで決定する
        threadgroup float local_max[THREADGROUP_SIZE];
        threadgroup uint  local_index[THREADGROUP_SIZE];
        float best = -1.0;
        uint best_row = i;
        for (uint r = i + tid; r < n_features; r += THREADGROUP_SIZE) {
            float val = fabs(A[r * (n_features + 1) + i]);
            if (val > best) {
                best = val;
                best_row = r;
            }
        }
        local_max[tid] = best;
        local_index[tid] = best_row;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // リダクション（固定サイズの小領域のみ使用）
        uint active = min((uint)THREADGROUP_SIZE, n_features - i);
        for (uint stride = active / 2; stride > 0; stride /= 2) {
            if (tid < stride && (tid + stride) < active) {
                if (local_max[tid] < local_max[tid + stride]) {
                    local_max[tid] = local_max[tid + stride];
                    local_index[tid] = local_index[tid + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint pivotRow = local_index[0];
        float pivotVal = A[pivotRow * (n_features + 1) + i];
        
        // ③ エラーチェック（ピボットが小さすぎる場合）
        if (fabs(pivotVal) < EPSILON) {
            if (tid == 0) {
                *error_flag = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (*error_flag != 0) {
            return;
        }
        
        // ④ 行スワップ（必要なら pivotRow と i 行目を全列で交換）
        if (tid == 0 && pivotRow != i) {
            for (uint c = 0; c < (n_features + 1); c++) {
                float tmp = A[i * (n_features + 1) + c];
                A[i * (n_features + 1) + c] = A[pivotRow * (n_features + 1) + c];
                A[pivotRow * (n_features + 1) + c] = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // ⑤ 消去処理（行 j = i+1 ～ n_features-1 の各行に対して）
        pivotVal = A[i * (n_features + 1) + i];  // 最新のピボット値
        for (uint j = i + 1 + tid; j < n_features; j += THREADGROUP_SIZE) {
            float multiplier = A[j * (n_features + 1) + i] / pivotVal;
            A[j * (n_features + 1) + i] = multiplier;  // L成分として保存
            for (uint k = i + 1; k < (n_features + 1); k++) {
                A[j * (n_features + 1) + k] -= multiplier * A[i * (n_features + 1) + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // --- 後退代入 ---
    // 拡大行列の右端の列に最終的な回帰係数が格納される
    if (tid == 0 && *error_flag == 0) {
        for (int i = int(n_features) - 1; i >= 0; i--) {
            float sum = A[i * (n_features + 1) + n_features];
            for (uint j = i + 1; j < n_features; j++) {
                sum -= A[i * (n_features + 1) + j] * A[j * (n_features + 1) + n_features];
            }
            A[i * (n_features + 1) + n_features] = sum / A[i * (n_features + 1) + i];
        }
    }
}
