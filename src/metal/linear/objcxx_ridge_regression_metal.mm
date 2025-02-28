#include "metal/linear/objcxx_ridge_regression_metal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#import <exception>
#import <iostream>
#include <omp.h>
#include <cblas.h>
#import <vector>

@interface RidgeRegressionMetalOBJCXX ()
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLComputePipelineState> pipelineState;
@property(nonatomic, strong) NSMutableArray<NSNumber *> *coefficients;
@end

@implementation RidgeRegressionMetalOBJCXX

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

    id<MTLFunction> function =
        [_library newFunctionWithName:@"ridge_regression_large"];
    _pipelineState = [_device newComputePipelineStateWithFunction:function
                                                            error:&error];
    if (!_pipelineState) {
      NSLog(@"Error creating pipeline state: %@", error);
      exit(1);
    }

    _coefficients = [[NSMutableArray alloc] init];
  }
  return self;
}

- (void)fitWithX:(const std::vector<float> &)X
               y:(const std::vector<float> &)y
        n_samples:(int)n_samples
       n_features:(int)n_features
       lambdaVal:(float)lambda {
  // ここで X は n_samples x n_features の行列（row-major）、y は n_samples のベクトル

  // OpenBLAS を使うために、cblas.h をインクルードしておく必要があります。
  // #include <cblas.h>

  // ① X^T * X の計算
  // 結果は n_features x n_features の行列 XtX に格納
  std::vector<float> XtX(n_features * n_features, 0.0f);
  cblas_sgemm(CblasRowMajor,    // データは row-major
              CblasTrans,       // X^T を使う
              CblasNoTrans,     // X はそのまま
              n_features,       // M: 出力行数 = n_features
              n_features,       // N: 出力列数 = n_features
              n_samples,        // K: 内部ループの回数 = n_samples
              1.0f,             // alpha
              X.data(),         // A: X (サイズ n_samples x n_features)
              n_features,       // lda: X の各行の要素数（row-major）
              X.data(),         // B: X (そのまま)
              n_features,       // ldb
              0.0f,             // beta
              XtX.data(),       // C: 結果行列 XtX
              n_features);      // ldc

  // ② X^T * y の計算
  // 結果は n_features のベクトル XtY に格納
  std::vector<float> XtY(n_features, 0.0f);
  cblas_sgemv(CblasRowMajor,    // X は row-major
              CblasTrans,       // X^T を使う
              n_samples,        // X の行数
              n_features,       // X の列数
              1.0f,             // alpha
              X.data(),         // X
              n_features,       // lda
              y.data(),         // y
              1,                // incx
              0.0f,             // beta
              XtY.data(),       // 結果ベクトル XtY
              1);               // incy

  // ③ 拡大行列 A_aug の生成
  // A_aug のサイズは n_features x (n_features + 1)
  // 左側 n_features×n_features に XtX を、右端の列に XtY を格納
  std::vector<float> A_aug(n_features * (n_features + 1), 0.0f);
  for (int i = 0; i < n_features; i++) {
    // X^T * X 部分のコピー
    for (int j = 0; j < n_features; j++) {
      A_aug[i * (n_features + 1) + j] = XtX[i * n_features + j];
    }
    // X^T * y 部分（最終列）
    A_aug[i * (n_features + 1) + n_features] = XtY[i];
  }

  // 以降は、生成した拡大行列 A_aug を Metal カーネルに渡して LU分解・後退代入を実行します。
  // A_aug のサイズは n_features x (n_features+1)
  // バッファ作成例：
  id<MTLBuffer> bufferA = [_device newBufferWithBytes:A_aug.data()
                                               length:A_aug.size() * sizeof(float)
                                              options:MTLResourceStorageModeShared];
  uint error_flg = 0;
  id<MTLBuffer> bufferError = [_device newBufferWithBytes:&error_flg
                                                   length:sizeof(uint)
                                                  options:MTLResourceStorageModeShared];
  // 定数バッファ（n_features, lambda）も作成
  id<MTLBuffer> bufferNFeatures = [_device newBufferWithBytes:&n_features
                                                       length:sizeof(uint)
                                                      options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferLambda = [_device newBufferWithBytes:&lambda
                                                    length:sizeof(float)
                                                   options:MTLResourceStorageModeShared];

  // Metal カーネルへの引数設定とディスパッチ
  id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  [encoder setComputePipelineState:_pipelineState];
  // バッファ割り当て: index 0 -> 拡大行列, 1 -> n_features, 2 -> lambda, 3 -> error_flag
  [encoder setBuffer:bufferA offset:0 atIndex:0];
  [encoder setBuffer:bufferNFeatures offset:0 atIndex:1];
  [encoder setBuffer:bufferLambda offset:0 atIndex:2];
  [encoder setBuffer:bufferError offset:0 atIndex:3];

  // 適切なスレッドグループサイズでディスパッチ
  MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
  MTLSize gridSize = MTLSizeMake(256, 1, 1);
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  memcpy(&error_flg, [bufferError contents], sizeof(uint));
  if (error_flg == 1) {
    std::runtime_error("LU decomposition failed.");
  }

  // カーネル実行後、回帰係数は拡大行列 A_aug の最終列に格納されているので取り出す
  float* A_data = (float*)[bufferA contents];
  std::vector<float> coeff(n_features);
  for (int i = 0; i < n_features; i++) {
    coeff[i] = A_data[i * (n_features + 1) + n_features];
  }

  [_coefficients removeAllObjects];
  for (int i = 0; i < n_features; i++) {
    [_coefficients addObject:@(coeff[i])];
  }
}

- (std::vector<float>)predictWithX:(const std::vector<float> &)X
                              rows:(int)rows
                              cols:(int)cols {
  std::vector<float> y_pred(rows, 0.0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      y_pred[i] += X[i * cols + j] * [_coefficients[j] floatValue];
    }
  }
  return y_pred;
}

@end
