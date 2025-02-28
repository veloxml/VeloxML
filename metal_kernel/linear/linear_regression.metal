#include <metal_stdlib>
using namespace metal;

#define MAX_SIZE 64

#define BLOCK_SIZE 16

#define THREADGROUP_SIZE 32

// LU分解（部分ピボット付き、エラーチェック・並列化あり）
// A: 行優先で格納された行列（サイズ: rows x cols）
// rows, cols: 行数・列数（通常、min(rows, cols) 回の反復を行う）
// error_flag: エラー発生時に1が書き込まれる（事前に0で初期化しておく）
kernel void lu_decomposition_optim2(device float*       A           [[ buffer(0) ]],
                             constant uint &     rows        [[ buffer(1) ]],
                             constant uint &     cols        [[ buffer(2) ]],
                             device uint*        error_flag  [[ buffer(3) ]],
                             uint                tid         [[ thread_position_in_threadgroup ]]) {

    // 有効な反復回数は、行数と列数の小さい方
    uint min_dim = (rows < cols) ? rows : cols;

    // 各ピボット反復ごとに処理
    for (uint i = 0; i < min_dim; i++) {

        // --- 部分ピボット選択 ---
        // i列の i行以降の中で、絶対値が最大の要素を持つ行を選ぶ
        // 各スレッドが候補を探索し、threadgroup内でリダクションする
        threadgroup float local_max[THREADGROUP_SIZE];
        threadgroup uint  local_index[THREADGROUP_SIZE];

        // 初期化：候補が存在しないスレッドは値0とする
        float best = 0.0;
        uint best_row = i;  // デフォルトは現在の行
        // 各スレッドは i 行以降のうち、tid 番目から THREADGROUP_SIZE 刻みで担当
        for (uint r = i + tid; r < rows; r += THREADGROUP_SIZE) {
            float val = fabs(A[r * cols + i]);
            if(val > best) {
                best = val;
                best_row = r;
            }
        }
        local_max[tid] = best;
        local_index[tid] = best_row;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // リダクション：threadgroup内で最大値と対応する行番号を求める
        uint stride = THREADGROUP_SIZE / 2;
        while (stride > 0) {
            if (tid < stride) {
                if (local_max[tid] < local_max[tid + stride]) {
                    local_max[tid] = local_max[tid + stride];
                    local_index[tid] = local_index[tid + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            stride /= 2;
        }

        // thread 0が決定したピボット行情報を持つ
        uint pivotRow = local_index[0];
        float pivotVal = A[pivotRow * cols + i];

        // --- エラーチェック ---
        // ピボット値が小さすぎる場合、特異行列としてエラー扱い
        if (fabs(pivotVal) < 1e-6) {
            if (tid == 0) {
                *error_flag = 1;
            }
            return;
        }

        // --- 行スワップ ---
        // 部分ピボットの場合、pivotRow と i行目が異なれば行全体をスワップする
        if (tid == 0 && pivotRow != i) {
            for (uint c = i; c < cols; c++) {
                float temp = A[i * cols + c];
                A[i * cols + c] = A[pivotRow * cols + c];
                A[pivotRow * cols + c] = temp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- 消去処理（LU分解の下三角部分の更新およびU部分の更新） ---
        // 各スレッドで i+1 行目以降を担当
        float pivot = A[i * cols + i]; // 更新後のピボット（スワップ済み）
        for (uint j = i + 1 + tid; j < rows; j += THREADGROUP_SIZE) {
            // L要素の更新
            A[j * cols + i] /= pivot;
            float multiplier = A[j * cols + i];
            // U部分の更新：i+1列以降
            for (uint k = i + 1; k < cols; k++) {
                A[j * cols + k] -= multiplier * A[i * cols + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}


kernel void lu_decomposition_optim(
    device float* A,     // 入力行列（サイズ: N×N，行優先）
    device float* L,     // 出力 L 行列（サイズ: N×N; 単位下三角）
    device float* U,     // 出力 U 行列（サイズ: N×N; 上三角）
    constant uint &N,    // 行列の次元 N
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
)
{
    // ブロック分割により，行列全体を対角ブロックを順次処理する
    for (uint k = 0; k < N; k += BLOCK_SIZE) {
        uint curBlockSize = min((uint)BLOCK_SIZE, N - k);

        // ────────────── Step 1: 対角ブロックの因子分解 ──────────────
        // 対角ブロック (A[k:k+curBlockSize, k:k+curBlockSize]) を threadgroup メモリにロード
        threadgroup float diagBlock[BLOCK_SIZE * BLOCK_SIZE];
        for (uint idx = tid; idx < curBlockSize * curBlockSize; idx += tg_size) {
            uint i = idx / curBlockSize;
            uint j = idx % curBlockSize;
            diagBlock[i * BLOCK_SIZE + j] = A[(k + i) * N + (k + j)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 対角ブロック上で，Doolittle のアルゴリズムで LU 分解を行う（シリアル処理）
        if (tid == 0) {
            for (uint i = 0; i < curBlockSize; i++) {
                // U 部分：j = i...curBlockSize-1
                for (uint j = i; j < curBlockSize; j++) {
                    float sum = 0.0;
                    for (uint l = 0; l < i; l++) {
                        sum += diagBlock[i * BLOCK_SIZE + l] * diagBlock[l * BLOCK_SIZE + j];
                    }
                    diagBlock[i * BLOCK_SIZE + j] = diagBlock[i * BLOCK_SIZE + j] - sum;
                }
                // L 部分：i2 = i+1...curBlockSize-1
                for (uint i2 = i + 1; i2 < curBlockSize; i2++) {
                    float sum = 0.0;
                    for (uint l = 0; l < i; l++) {
                        sum += diagBlock[i2 * BLOCK_SIZE + l] * diagBlock[l * BLOCK_SIZE + i];
                    }
                    diagBlock[i2 * BLOCK_SIZE + i] = (diagBlock[i2 * BLOCK_SIZE + i] - sum) / diagBlock[i * BLOCK_SIZE + i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 対角ブロックの結果をグローバルメモリへ書き出す（L は単位下三角、U は上三角）
        for (uint idx = tid; idx < curBlockSize * curBlockSize; idx += tg_size) {
            uint i = idx / curBlockSize;
            uint j = idx % curBlockSize;
            uint global_i = k + i;
            uint global_j = k + j;
            if (i > j) {
                L[global_i * N + global_j] = diagBlock[i * BLOCK_SIZE + j];
                U[global_i * N + global_j] = 0.0;
            } else if (i == j) {
                L[global_i * N + global_j] = 1.0;
                U[global_i * N + global_j] = diagBlock[i * BLOCK_SIZE + j];
            } else {
                L[global_i * N + global_j] = 0.0;
                U[global_i * N + global_j] = diagBlock[i * BLOCK_SIZE + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ────────────── Step 2: 対角ブロックの右側ブロック（行方向）の更新 ──────────────
        // U 部分の更新：各ブロック j (j >= k+curBlockSize)
        for (uint j = k + curBlockSize; j < N; j += BLOCK_SIZE) {
            uint curBlockWidth = min((uint)BLOCK_SIZE, N - j);
            // 対角ブロックの行に対応する A の部分を rowBlock としてロード
            threadgroup float rowBlock[BLOCK_SIZE * BLOCK_SIZE]; // 寸法: curBlockSize × curBlockWidth
            for (uint idx = tid; idx < curBlockSize * curBlockWidth; idx += tg_size) {
                uint i = idx / curBlockWidth;
                uint jj = idx % curBlockWidth;
                rowBlock[i * BLOCK_SIZE + jj] = A[(k + i) * N + (j + jj)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // 前進代入により，diagBlock の L 部分を使って rowBlock を更新
            if (tid == 0) {
                for (uint i = 0; i < curBlockSize; i++) {
                    for (uint jj = 0; jj < curBlockWidth; jj++) {
                        float sum = 0.0;
                        for (uint l = 0; l < i; l++) {
                            sum += diagBlock[i * BLOCK_SIZE + l] * rowBlock[l * BLOCK_SIZE + jj];
                        }
                        rowBlock[i * BLOCK_SIZE + jj] = rowBlock[i * BLOCK_SIZE + jj] - sum;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // 更新結果を U の該当部分に書き出す
            for (uint idx = tid; idx < curBlockSize * curBlockWidth; idx += tg_size) {
                uint i = idx / curBlockWidth;
                uint jj = idx % curBlockWidth;
                U[(k + i) * N + (j + jj)] = rowBlock[i * BLOCK_SIZE + jj];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ────────────── Step 3: 対角ブロックの下側ブロック（列方向）の更新 ──────────────
        // L 部分の更新：各ブロック i (i >= k+curBlockSize)
        for (uint i = k + curBlockSize; i < N; i += BLOCK_SIZE) {
            uint curBlockHeight = min((uint)BLOCK_SIZE, N - i);
            threadgroup float colBlock[BLOCK_SIZE * BLOCK_SIZE]; // 寸法: curBlockHeight × curBlockSize
            for (uint idx = tid; idx < curBlockHeight * curBlockSize; idx += tg_size) {
                uint r = idx / curBlockSize;
                uint c = idx % curBlockSize;
                colBlock[r * BLOCK_SIZE + c] = A[(i + r) * N + (k + c)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // 後退代入により，diagBlock の U 部分を用いて colBlock を更新
            if (tid == 0) {
                for (uint r = 0; r < curBlockHeight; r++) {
                    for (uint c = 0; c < curBlockSize; c++) {
                        float sum = 0.0;
                        for (uint l = 0; l < c; l++) {
                            sum += colBlock[r * BLOCK_SIZE + l] * diagBlock[l * BLOCK_SIZE + c];
                        }
                        colBlock[r * BLOCK_SIZE + c] = (colBlock[r * BLOCK_SIZE + c] - sum) / diagBlock[c * BLOCK_SIZE + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // 書き出し：更新された colBlock を L の該当部分に書く
            for (uint idx = tid; idx < curBlockHeight * curBlockSize; idx += tg_size) {
                uint r = idx / curBlockSize;
                uint c = idx % curBlockSize;
                L[(i + r) * N + (k + c)] = colBlock[r * BLOCK_SIZE + c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ────────────── Step 4: 残余部分（右下の trailing submatrix）の更新 ──────────────
        for (uint i = k + curBlockSize; i < N; i += BLOCK_SIZE) {
            uint curBlockHeight = min((uint)BLOCK_SIZE, N - i);
            for (uint j = k + curBlockSize; j < N; j += BLOCK_SIZE) {
                uint curBlockWidth = min((uint)BLOCK_SIZE, N - j);
                threadgroup float updateBlock[BLOCK_SIZE * BLOCK_SIZE]; // 寸法: curBlockHeight × curBlockWidth
                for (uint idx = tid; idx < curBlockHeight * curBlockWidth; idx += tg_size) {
                    uint r = idx / curBlockWidth;
                    uint c = idx % curBlockWidth;
                    updateBlock[r * BLOCK_SIZE + c] = A[(i + r) * N + (j + c)];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // 更新：updateBlock -= L[i, k:k+curBlockSize] * U[k:k+curBlockSize, j]
                for (uint l = 0; l < curBlockSize; l++) {
                    for (uint idx = tid; idx < curBlockHeight * curBlockWidth; idx += tg_size) {
                        uint r = idx / curBlockWidth;
                        uint c = idx % curBlockWidth;
                        updateBlock[r * BLOCK_SIZE + c] -= L[(i + r) * N + (k + l)] * U[(k + l) * N + (j + c)];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                // 書き出し：更新された updateBlock をグローバルメモリの A に書き戻す
                for (uint idx = tid; idx < curBlockHeight * curBlockWidth; idx += tg_size) {
                    uint r = idx / curBlockWidth;
                    uint c = idx % curBlockWidth;
                    A[(i + r) * N + (j + c)] = updateBlock[r * BLOCK_SIZE + c];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // 最終的に，グローバルメモリの L, U に対角ブロックおよび更新結果が反映される．
}



kernel void lu_decomposition(
    device float* A, // 行列 (サイズ N * N)
    device float* L, // L成分 (サイズ N * N)
    device float* U, // U成分 (サイズ N * N)
    constant uint& N,
    uint gid [[thread_position_in_grid]]) 
{
    int i = gid;
    if (i >= N) return;

    for (int j = i; j < N; j++) {
        U[i * N + j] = A[i * N + j];
        for (int k = 0; k < i; k++) {
            U[i * N + j] -= L[i * N + k] * U[k * N + j];
        }
    }
    
    for (int j = i; j < N; j++) {
        if (i == j)
            L[i * N + i] = 1.0;
        else {
            L[j * N + i] = A[j * N + i];
            for (int k = 0; k < i; k++) {
                L[j * N + i] -= L[j * N + k] * U[k * N + i];
            }
            L[j * N + i] /= U[i * N + i];
        }
    }
}
