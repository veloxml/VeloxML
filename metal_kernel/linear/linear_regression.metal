#include <metal_stdlib>
using namespace metal;

#define MAX_SIZE 64

#define BLOCK_SIZE 16

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
