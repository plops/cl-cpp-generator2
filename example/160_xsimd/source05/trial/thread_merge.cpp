#include <iostream>
#include <vector>
#include <numeric> // For std::iota
#include <algorithm> // For std::merge
#include <chrono> // For std::chrono_literals

// g++ -std=gnu++20 -march=native -O2 -ggdb3 thread_merge.cpp -o thread_merge

// Function to compute a vector of 5 elements
std::vector<int> computeVector() {
    std::vector<int> result(5);
    std::iota(result.begin(), result.end(), 0); // Fill with sequential values starting from 0
    return result;
}

int main() {
    const int num_threads = 6;
    std::vector<std::jthread> threads(num_threads);
    std::vector<std::vector<int>> results(num_threads);

    // Launch the computation in each thread
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::jthread([i, &results]() mutable {
            results[i] = computeVector();
        });
    }

    // Join all threads to ensure they finish
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Merge the results into one vector with 30 elements
    std::vector<int> final_result(30);
    int pos = 0;
    for (const auto& vec : results) {
        std::copy(vec.begin(), vec.end(), final_result.begin() + pos);
        pos += vec.size();
    }

    // Print the merged result
    for (int val : final_result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
