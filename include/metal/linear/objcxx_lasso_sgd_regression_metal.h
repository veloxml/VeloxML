#ifndef OBJ_CXX_LASSO_SGD_REGRESSION_METAL_H
#define OBJ_CXX_LASSO_SGD_REGRESSION_METAL_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <vector>

NS_ASSUME_NONNULL_BEGIN

@interface LassoSGDMetalOBJCXX : NSObject

- (instancetype)init;
- (void)fitWithX:(const std::vector<float> &)X 
               y:(const std::vector<float> &)y 
            rows:(int)rows 
            cols:(int)cols 
        lambda:(float)lambda 
        lr:(float)lr 
        epochs:(int)epochs 
        batch_size:(int)batch_size;
- (std::vector<float>)predictWithX:(const std::vector<float> &)X 
                              rows:(int)rows 
                              cols:(int)cols;

@end

NS_ASSUME_NONNULL_END

#endif /* OBJ_CXX_LASSO_SGD_REGRESSION_METAL_H */
