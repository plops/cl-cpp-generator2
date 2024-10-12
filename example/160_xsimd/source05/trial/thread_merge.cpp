#include <iostream>
#include <vector>
#include <numeric> // For iota
#include <algorithm> // For merge
#include <chrono> // For chrono_literals
#include <thread>

using namespace std;

// g++ -std=gnu++20 -march=native -O2 -ggdb3 thread_merge.cpp -o thread_merge

// Function to compute a vector of 5 elements
vector<int> computeVector(int i) {
    vector<int> result(5);
    iota(result.begin(), result.end(), i*6); // Fill with sequential values starting from 0
    return result;
}

int main() {
    const int num_threads = 6;
    vector<jthread> threads(num_threads);
    vector<vector<int>> results(num_threads);

    // Launch the computation in each thread
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = jthread([i, &results]() mutable {
            results[i] = computeVector(i);
        });
    }

    // Join all threads to ensure they finish
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Merge the results into one vector with 30 elements
    vector<int> final_result(30);
    int pos = 0;
    for (const auto& vec : results) {
        copy(vec.begin(), vec.end(), final_result.begin() + pos);
        pos += vec.size();
    }

    // Print the merged result
    for (int val : final_result) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
