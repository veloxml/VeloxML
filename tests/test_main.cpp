#include <gtest/gtest.h>

// Google Test のメイン関数
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "SVMClassificationTest*";
    return RUN_ALL_TESTS();
}