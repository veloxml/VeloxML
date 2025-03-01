#import "metal/linear/objcxx_lasso_regression_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <cblas.h>
#import <cmath>
#import <dispatch/dispatch.h>
#import <lapacke.h>
#import <omp.h>
#import <vector>

// CPU側で Lipschitz 定数を計算する関数（X は row-major の float 配列）
double computeLipschitzConstant(const std::vector<float> &X, int n, int m) {
  std::vector<double> Xd(n * m);
#pragma omp parallel for
  for (int i = 0; i < n * m; i++) {
    Xd[i] = static_cast<double>(X[i]);
  }
  std::vector<double> A(m * m, 0.0);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, n, 1.0, Xd.data(),
              n, Xd.data(), n, 0.0, A.data(), m);
  std::vector<double> eigvals(m, 0.0);
  std::vector<double> A_copy = A;
  int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'U', m, A_copy.data(), m,
                           eigvals.data());
  if (info != 0)
    @throw [NSException exceptionWithName:@"LipschitzException"
                                   reason:@"Eigenvalue computation failed in "
                                          @"computeLipschitzConstant"
                                 userInfo:nil];
  return *std::max_element(eigvals.begin(), eigvals.end());
}

@interface LassoRegressionMetalOBJCXX ()
// Metal 関連プロパティ
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property(nonatomic, strong) id<MTLLibrary> library;
// それぞれのカーネル用パイプライン
@property(nonatomic, strong) id<MTLComputePipelineState>
    pipelineStateComputePred;
@property(nonatomic, strong) id<MTLComputePipelineState> pipelineStateUpdate;
// 結果の係数を保持するプロパティ
@property(nonatomic, strong) NSMutableArray<NSNumber *> *coefficients;
@end

@implementation LassoRegressionMetalOBJCXX

- (instancetype)init {
  self = [super init];
  if (self) {
    _device = MTLCreateSystemDefaultDevice();
    _commandQueue = [_device newCommandQueue];

    NSError *error = nil;
    NSString *kernelPath = @"kernel/shaders.metallib";
    NSData *kernelData = [NSData dataWithContentsOfFile:kernelPath];
    if (!kernelData) {
      NSLog(@"Failed to load Metal kernel file at %@", kernelPath);
      exit(1);
    }
    dispatch_data_t metalData =
        dispatch_data_create(kernelData.bytes, kernelData.length,
                             DISPATCH_DATA_DESTRUCTOR_DEFAULT, NULL);
    _library = [_device newLibraryWithData:metalData error:&error];
    if (!_library) {
      NSLog(@"Error loading Metal library: %@", error);
      exit(1);
    }

    // compute_pred カーネルのパイプライン作成
    id<MTLFunction> computePredFunction =
        [_library newFunctionWithName:@"compute_pred"];
    _pipelineStateComputePred =
        [_device newComputePipelineStateWithFunction:computePredFunction
                                               error:&error];
    if (!_pipelineStateComputePred) {
      NSLog(@"Error creating compute_pred pipeline state: %@", error);
      exit(1);
    }

    // fista_lasso_update カーネルのパイプライン作成
    id<MTLFunction> updateFunction =
        [_library newFunctionWithName:@"fista_lasso_update"];
    _pipelineStateUpdate =
        [_device newComputePipelineStateWithFunction:updateFunction
                                               error:&error];
    if (!_pipelineStateUpdate) {
      NSLog(@"Error creating fista_lasso_update pipeline state: %@", error);
      exit(1);
    }

    _coefficients = [[NSMutableArray alloc] init];
  }
  return self;
}

- (void)fitWithX:(const std::vector<float> &)X
               y:(const std::vector<float> &)y
            rows:(int)rows
            cols:(int)cols
          lambda:(float)lambda
              lr:(float)lr // 初期値; 後で Lipschitz 定数により上書き
        max_iter:(int)max_iter
             tol:(float)tol {
  int size = cols;
  // ホスト側の変数初期化
  std::vector<float> w(size, 0.0f);
  std::vector<float> w_prev(size, 0.0f);
  std::vector<float> z(size, 0.0f);
  float t = 1.0f;

  // Metal バッファの作成
  id<MTLBuffer> bufferX =
      [_device newBufferWithBytes:X.data()
                           length:sizeof(float) * rows * cols
                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferY =
      [_device newBufferWithBytes:y.data()
                           length:sizeof(float) * rows
                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferW =
      [_device newBufferWithBytes:w.data()
                           length:sizeof(float) * size
                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferWPrev =
      [_device newBufferWithBytes:w_prev.data()
                           length:sizeof(float) * size
                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferZ =
      [_device newBufferWithBytes:z.data()
                           length:sizeof(float) * size
                          options:MTLResourceStorageModeShared];
  // 予測値を保存するバッファ (rows 個)
  id<MTLBuffer> bufferPred =
      [_device newBufferWithLength:sizeof(float) * rows
                           options:MTLResourceStorageModeShared];
  // t 用バッファ（長さ 1）
  id<MTLBuffer> bufferT =
      [_device newBufferWithBytes:&t
                           length:sizeof(float)
                          options:MTLResourceStorageModeShared];

  // 初期値の設定（z, w_prev は 0 で初期化済み）
  memcpy([bufferZ contents], z.data(), sizeof(float) * size);
  memcpy([bufferWPrev contents], w_prev.data(), sizeof(float) * size);

  // CPU 側で Lipschitz 定数を計算し、lr を 1 / L_const に設定
  double L_const = computeLipschitzConstant(X, rows, cols);
  lr = 1.0f / static_cast<float>(L_const);
  NSLog(@"Computed Lipschitz constant: %f, setting lr = %f", L_const, lr);

  // 反復処理
  for (int iter = 0; iter < max_iter; iter++) {
    // --- ① 予測値計算カーネル compute_pred ---
    {
      id<MTLCommandBuffer> cmdBuf = [_commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:_pipelineStateComputePred];
      [encoder setBuffer:bufferX offset:0 atIndex:0];
      [encoder setBuffer:bufferW offset:0 atIndex:1];
      [encoder setBuffer:bufferPred offset:0 atIndex:2];
      [encoder setBytes:&rows length:sizeof(uint) atIndex:3];
      [encoder setBytes:&cols length:sizeof(uint) atIndex:4];

      MTLSize gridSize = MTLSizeMake(rows, 1, 1);
      MTLSize threadGroupSize = MTLSizeMake(MIN(rows, 64), 1, 1);
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // --- ② 勾配計算＋重み更新カーネル fista_lasso_update ---
    {
      id<MTLCommandBuffer> cmdBuf = [_commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:_pipelineStateUpdate];
      [encoder setBuffer:bufferX offset:0 atIndex:0];
      [encoder setBuffer:bufferY offset:0 atIndex:1];
      [encoder setBuffer:bufferPred offset:0 atIndex:2];
      [encoder setBuffer:bufferW offset:0 atIndex:3];
      [encoder setBuffer:bufferWPrev offset:0 atIndex:4];
      [encoder setBuffer:bufferZ offset:0 atIndex:5];
      [encoder setBytes:&lambda length:sizeof(float) atIndex:6];
      [encoder setBytes:&lr length:sizeof(float) atIndex:7];
      [encoder setBytes:&rows length:sizeof(uint) atIndex:8];
      [encoder setBytes:&cols length:sizeof(uint) atIndex:9];

      MTLSize gridSize = MTLSizeMake(size, 1, 1);
      MTLSize threadGroupSize = MTLSizeMake(MIN(size, 16), 1, 1);
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // --- ③ ホスト側で加速更新 & 収束判定 ---
    {
      // コピー: GPU 上の更新済み w をホストに取り込む
      memcpy(w.data(), [bufferW contents], sizeof(float) * size);
      memcpy(w_prev.data(), [bufferWPrev contents], sizeof(float) * size);

      // FISTA 加速更新: t, z の更新
      float t_new = (1.0f + sqrtf(1.0f + 4.0f * t * t)) / 2.0f;
      float momentum = (t - 1.0f) / t_new;

#pragma omp parallel for
      for (int i = 0; i < size; i++) {
        // 更新 z
        z[i] = w[i] + momentum * (w[i] - w_prev[i]);
      }

      float diff_norm = 0.0f;
#pragma omp parallel for reduction(+ : diff_norm)
      for (int i = 0; i < size; i++) {
        float diff = w[i] - w_prev[i];
        diff_norm += diff * diff;
      }
      diff_norm = sqrtf(diff_norm);

      // 更新 t
      t = t_new;

      // 収束判定: tol 以下なら終了
      if (diff_norm < tol) {
        NSLog(@"Converged at iteration %d, norm diff = %f", iter, diff_norm);
        break;
      }

      // GPU バッファへ反映
      memcpy([bufferZ contents], z.data(), sizeof(float) * size);
      memcpy([bufferWPrev contents], w.data(), sizeof(float) * size);
      memcpy([bufferT contents], &t, sizeof(float));
    }
  }

  // --- 結果取得 ---
  memcpy(w.data(), [bufferW contents], sizeof(float) * size);
  [_coefficients removeAllObjects];
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    [_coefficients addObject:@(w[i])];
  }
}

- (std::vector<float>)predictWithX:(const std::vector<float> &)X
                              rows:(int)rows
                              cols:(int)cols {
  std::vector<float> y_pred(rows, 0.0f);
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
#pragma omp parallel for reduction(+ : i)
    for (int j = 0; j < cols; ++j) {
      y_pred[i] += X[i * cols + j] * [_coefficients[j] floatValue];
    }
  }
  return y_pred;
}

@end
