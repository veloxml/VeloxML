#include "metal/linear/objcxx_linear_regression_metal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h> // dispatch_data_create を使うために追加
#import <iostream>
#import <vector>
#import <omp.h>

@interface LinearRegressionMetalOBJCXX ()
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLComputePipelineState> pipelineState;
@property(nonatomic, strong) NSMutableArray<NSNumber *> *coefficients;
@end

@implementation LinearRegressionMetalOBJCXX

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
    // NSData から dispatch_data_t に変換する
    dispatch_data_t metalData =
        dispatch_data_create(kernelData.bytes, kernelData.length,
                             DISPATCH_DATA_DESTRUCTOR_DEFAULT, NULL);
    _library = [_device newLibraryWithData:metalData error:&error];
    if (!_library) {
      NSLog(@"Error loading Metal library: %@", error);
      exit(1);
    }

    id<MTLFunction> function =
        [_library newFunctionWithName:@"lu_decomposition_optim"];
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

- (void)dealloc {
  // _device = nil;
  // _commandQueue = nil;
  // _library = nil;
  // _pipelineState = nil;
  [super dealloc];
}

- (void)fitWithX:(const std::vector<float> &)X
               y:(const std::vector<float> &)y
            rows:(int)rows
            cols:(int)cols {
  int N = cols + 1;
  std::vector<float> A(rows * N, 0.0);
  std::vector<float> L(rows * N, 0.0);
  std::vector<float> U(rows * N, 0.0);

#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
#pragma omp parallel for
    for (int j = 0; j < cols; ++j) {
      A[i * N + j] = X[i * cols + j];
    }
    A[i * N + cols] = y[i];
  }

  id<MTLBuffer> bufferA =
      [_device newBufferWithBytes:A.data()
                           length:N * N * sizeof(float)
                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferL =
      [_device newBufferWithLength:N * N * sizeof(float)
                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferU =
      [_device newBufferWithLength:N * N * sizeof(float)
                           options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

  [encoder setComputePipelineState:_pipelineState];
  [encoder setBuffer:bufferA offset:0 atIndex:0];
  [encoder setBuffer:bufferL offset:0 atIndex:1];
  [encoder setBuffer:bufferU offset:0 atIndex:2];
  [encoder setBytes:&N length:sizeof(uint) atIndex:3];

  MTLSize gridSize = MTLSizeMake(N, 1, 1);
  MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
  [encoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  memcpy(L.data(), [bufferL contents], N * N * sizeof(float));
  memcpy(U.data(), [bufferU contents], N * N * sizeof(float));

  [_coefficients removeAllObjects];
  for (int i = 0; i < N; i++) {
    [_coefficients addObject:@(L[i * N + N - 1])];
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
