#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fmt/core.h>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <thread>
uint32_t        buf_size, numProducers, numConsumers, *buffer, fillIndex, useIndex, count = 0;
pthread_cond_t  modify;
pthread_mutex_t mutex;
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
// This function is generates log output including wall clock time, source file and line, function and optionally some
// variables that will be submitted as strings in an initializer_list. Arbitrary values are converted to strings using
// fmt::format

void lprint(std::string msg, std::initializer_list<std::string> il, std::string func, std::string file, int line) {
    std::chrono::duration<double> timestamp = std::chrono::high_resolution_clock::now() - g_start_time;
    std::cout << std::setw(10) << timestamp.count() << " " << std::this_thread::get_id() << " " << file << ":" << line
              << " " << func << " " << msg << " ";
    for (const auto& elem : il) { std::cout << elem; }
    std::cout << std::endl << std::flush;
}

// constant color strings
// for use in producer (a colored p):
#define PRED "\033[1;31mp\033[0m"
#define PGREEN "\033[1;32mp\033[0m"
#define PYELLOW "\033[1;33mp\033[0m"
#define PBLUE "\033[1;34mp\033[0m"
#define PMAGENTA "\033[1;35mp\033[0m"
#define PCYAN "\033[1;36mp\033[0m"
#define PWHITE "\033[1;37mp\033[0m"
// for use in consumer (a colored c followed by an id):
#define RED "\033[1;31mc%01d\033[0m"
#define GREEN "\033[1;32mc%01d\033[0m"
#define YELLOW "\033[1;33mc%01d\033[0m"
#define BLUE "\033[1;34mc%01d\033[0m"
#define MAGENTA "\033[1;35mc%01d\033[0m"
#define CYAN "\033[1;36mc%01d\033[0m"
#define WHITE "\033[1;37mc%01d\033[0m"
// functions append and head that will run concurrently

void append(uint32_t value) {
    lprint("", {" value='", fmt::format("{}", value), "'"}, __func__, __FILE__, __LINE__);
    buffer[fillIndex] = value;
    fillIndex         = ((fillIndex + 1) % buf_size);
    count++;
}


uint32_t head() {
    auto tmp{buffer[useIndex]};
    lprint("", {" useIndex='", fmt::format("{}", useIndex), "'", " tmp='", fmt::format("{}", tmp), "'"}, __func__,
           __FILE__, __LINE__);
    useIndex = ((useIndex + 1) % buf_size);
    count--;
    return tmp;
}


void* producer(void* arg) {
    while (1) {
        pthread_mutex_lock(&mutex);
        while (count == buf_size) { pthread_cond_wait(&modify, &mutex); }
        append(rand() % 10);
        pthread_cond_signal(&modify);
        pthread_mutex_unlock(&mutex);
    }
}


void* consumer(void* arg) {
    auto id{*static_cast<uint32_t*>(arg)};
    while (1) {
        pthread_mutex_lock(&mutex);
        while (0 == count) { pthread_cond_wait(&modify, &mutex); }
    }
    head();
    pthread_cond_signal(&modify);
    pthread_mutex_unlock(&mutex);
}


int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: ./consumer_producer <buffer_size> <#producers> <#consumers>\n");
        printf("./consumer_producer 1 2 1  => deadlock possible\n");
        exit(1);
    }
    g_start_time = std::chrono::high_resolution_clock::now();
    lprint("start", {" argc='", fmt::format("{}", argc), "'", " (argv)[(0)]='", fmt::format("{}", argv[0]), "'"},
           __func__, __FILE__, __LINE__);
    srand(999);
    buf_size     = atoi(argv[1]);
    numProducers = atoi(argv[2]);
    numConsumers = atoi(argv[3]);
    lprint("initiate mutex and condition variable", {}, __func__, __FILE__, __LINE__);
    pthread_mutex_init(&mutex, nullptr);
    pthread_cond_init(&modify, nullptr);
    lprint("allocate buffer", {" buf_size='", fmt::format("{}", buf_size), "'"}, __func__, __FILE__, __LINE__);
    buffer = static_cast<uint32_t*>(malloc(buf_size * sizeof(uint32_t)));
    pthread_t prods[numProducers], cons[numConsumers];
    uint32_t  threadIds[numConsumers], i;
    lprint("start consumers", {" numConsumers='", fmt::format("{}", numConsumers), "'"}, __func__, __FILE__, __LINE__);
    for (decltype(0 + numConsumers + 1) i = 0; i < numConsumers; i += 1) {
        threadIds[i] = i;
        pthread_create(cons + i, nullptr, consumer, threadIds + i);
    }
    lprint("start producers", {" numProducers='", fmt::format("{}", numProducers), "'"}, __func__, __FILE__, __LINE__);
    for (decltype(0 + numProducers + 1) i = 0; i < numProducers; i += 1) {
        pthread_create(prods + i, nullptr, producer, nullptr);
    }
    lprint("wait for threads to finish", {}, __func__, __FILE__, __LINE__);
    for (decltype(0 + numProducers + 1) i = 0; i < numProducers; i += 1) { pthread_join(prods[i], nullptr); }
    for (decltype(0 + numConsumers + 1) i = 0; i < numConsumers; i += 1) { pthread_join(cons[i], nullptr); }
    lprint("leave program", {}, __func__, __FILE__, __LINE__);
    return 0;
}
