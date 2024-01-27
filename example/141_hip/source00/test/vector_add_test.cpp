#include "vector_add.hip.cpp"  
#include <gtest/gtest.h>  
  
// Test fixture for vector addition  
class VectorAddTest : public ::testing::Test {  
protected:  
    std::vector<float> A;  
    std::vector<float> B;  
    std::vector<float> expected;  
  
    void SetUp() override {  
        A.resize(N);  
        B.resize(N);  
        expected.resize(N);  
  
        for (int i = 0; i < N; i++) {  
            A[i] = static_cast<float>(i);  
            B[i] = static_cast<float>(i * 2);  
            expected[i] = A[i] + B[i];  
        }  
    }  
};  
  
// Test case to verify vector addition  
TEST_F(VectorAddTest, AddsVectorsCorrectly) {  
    auto C = runVectorAdd(A, B);  
    for (size_t i = 0; i < N; i++) {  
        EXPECT_NEAR(expected[i], C[i], 1e-5) << "Vectors differ at index " << i;  
    }  
}  
  
int main(int argc, char **argv) {  
    ::testing::InitGoogleTest(&argc, argv);  
    return RUN_ALL_TESTS();  
}  
