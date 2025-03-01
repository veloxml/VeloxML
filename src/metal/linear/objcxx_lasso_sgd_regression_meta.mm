#import "metal/linear/objcxx_lasso_sgd_regression_metal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#import <iostream>
#import <vector>

@interface LassoSGDMetalOBJCXX ()
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLComputePipelineState> pipelineState;
@property(nonatomic, strong) NSMutableArray<NSNumber *> *coefficients;
@property(nonatomic, strong) id<MTLComputePipelineState> predictPipelineState;
@end

@implementation LassoSGDMetalOBJCXX

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

    // 訓練用カーネル "lasso_sgd" のパイプライン状態を生成
    id<MTLFunction> function = [_library newFunctionWithName:@"lasso_sgd"];
    _pipelineState = [_device newComputePipelineStateWithFunction:function
                                                            error:&error];
    if (!_pipelineState) {
      NSLog(@"Error creating pipeline state: %@", error);
      exit(1);
    }

    // 予測処理用カーネル "predict_kernel"
    // のパイプライン状態を生成（GPUによる予測を有効化）
    id<MTLFunction> predictFunction =
        [_library newFunctionWithName:@"predict_kernel"];
    if (predictFunction) {
      _predictPipelineState =
          [_device newComputePipelineStateWithFunction:predictFunction
                                                 error:&error];
      if (!_predictPipelineState) {
        NSLog(@"Error creating prediction pipeline state: %@", error);
        // 必要に応じてCPU予測へフォールバック
      }
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
              lr:(float)lr
          epochs:(int)epochs
      batch_size:(int)batch_size {
  // 入力データと重み用のバッファを作成（バッファ再利用によりオーバーヘッド削減）
  id<MTLBuffer> bufferX =
      [_device newBufferWithBytes:X.data()
                           length:sizeof(float) * rows * cols
                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferY =
      [_device newBufferWithBytes:y.data()
                           length:sizeof(float) * rows
                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferW =
      [_device newBufferWithLength:sizeof(float) * cols
                           options:MTLResourceStorageModeShared];

  std::vector<float> w(cols, 0.0f);
  for (int i = 0; i < cols; i++) {
    w[i] = (float)rand() / RAND_MAX * 0.01f; // 0 ではなく、小さい値を持たせる
  }
  memcpy([bufferW contents], w.data(), sizeof(float) * cols);

  // メモリサイズの計算
  //   NSUInteger shared_memory_size = 32 * cols * sizeof(float);

  // エポック毎に、各バッチ更新を1つのコマンドバッファにまとめる
  for (int epoch = 0; epoch < epochs; epoch++) {
    for (int i = 0; i < rows; i += batch_size) {
      int current_batch = std::min(batch_size, rows - i);

      id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      [encoder setComputePipelineState:_pipelineState];
      // 各バッチのオフセット設定
      [encoder setBuffer:bufferX offset:i * cols * sizeof(float) atIndex:0];
      [encoder setBuffer:bufferY offset:i * sizeof(float) atIndex:1];
      [encoder setBuffer:bufferW offset:0 atIndex:2];
      [encoder setBytes:&lambda length:sizeof(float) atIndex:3];
      [encoder setBytes:&lr length:sizeof(float) atIndex:4];
      [encoder setBytes:&current_batch length:sizeof(uint) atIndex:5];
      [encoder setBytes:&cols length:sizeof(uint) atIndex:6];
      //   [encoder setThreadgroupMemoryLength:shared_memory_size atIndex:10];

      // グリッドサイズとスレッドグループサイズ（ここはGPUの特性に合わせて調整可能）
      // **スレッドグループとスレッド数の設定**
      NSUInteger threadsPerGrid = cols; // 特徴量の数だけスレッドを作成
      NSUInteger threadsPerThreadgroup =
          _pipelineState
              .maxTotalThreadsPerThreadgroup; // Metalの最適なスレッド数を取得

      MTLSize gridSize = MTLSizeMake(threadsPerGrid, 1, 1);
      MTLSize threadGroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
    }
    // std::cout << "Epoch: " << epoch << " is done." << std::endl;
  }

  // GPU上の重み更新結果を取得
  memcpy(w.data(), [bufferW contents], sizeof(float) * cols);

  [_coefficients removeAllObjects];
#pragma omp parallel for
  for (int i = 0; i < cols; i++) {
    [_coefficients addObject:@(w[i])];
  }
}

- (std::vector<float>)predictWithX:(const std::vector<float> &)X
                              rows:(int)rows
                              cols:(int)cols {
  // GPU用の予測カーネルが存在する場合はGPUで計算
  if (_predictPipelineState) {
    id<MTLBuffer> bufferX =
        [_device newBufferWithBytes:X.data()
                             length:sizeof(float) * rows * cols
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferW =
        [_device newBufferWithLength:sizeof(float) * cols
                             options:MTLResourceStorageModeShared];
    // 現在の重みをバッファに転送
    std::vector<float> w(cols, 0.0f);
    for (int i = 0; i < cols; i++) {
      w[i] = [_coefficients[i] floatValue];
    }
    memcpy([bufferW contents], w.data(), sizeof(float) * cols);

    id<MTLBuffer> bufferPred =
        [_device newBufferWithLength:sizeof(float) * rows
                             options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_predictPipelineState];
    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferW offset:0 atIndex:1];
    [encoder setBuffer:bufferPred offset:0 atIndex:2];
    [encoder setBytes:&cols length:sizeof(uint) atIndex:3];
    [encoder setBytes:&rows length:sizeof(uint) atIndex:4];

    MTLSize gridSize = MTLSizeMake(rows, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(16, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    float *predData = (float *)[bufferPred contents];
    std::vector<float> y_pred(predData, predData + rows);
    return y_pred;
  } else {
    // GPUの予測カーネルが利用できない場合は従来のCPU並列実装を利用
    std::vector<float> y_pred(rows, 0.0f);
#pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
      float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
      for (int j = 0; j < cols; ++j) {
        sum += X[i * cols + j] * [_coefficients[j] floatValue];
      }
      y_pred[i] = sum;
    }
    return y_pred;
  }
}

@end
