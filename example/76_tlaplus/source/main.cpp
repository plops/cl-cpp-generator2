#include <cassert>
#include <chrono>
#include <cmath>
#include <fmt/core.h>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
uint32_t buf_size, numProducers, numConsumers, *buffer, fillIndex, useIndex,
    count = 0;
pthread_cond_t modify;
pthread_mutex_t mutex;
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
void lprint(std::initializer_list<std::string> il) {
  std::chrono::duration<double> timestamp =
      std::chrono::high_resolution_clock::now() - g_start_time;
  (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
              << (std::this_thread::get_id()) << (" ");
  for (const auto &elem : il) {
    (std::cout) << (elem);
  }
  (std::cout) << (std::endl) << (std::flush);
}
void append(uint32_t value) {
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ", "append",
          " ", " value='", fmt::format("{}", value), "'"});
  buffer[fillIndex] = value;
  fillIndex = ((fillIndex) + (1)) % buf_size;
  (count)++;
}
uint32_t head() {
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ", "head",
          " "});
  auto tmp = buffer[useIndex];
  useIndex = ((useIndex) + (1)) % buf_size;
  (count)--;
  return tmp;
}
void *producer(void *arg) {
#define PRED "\033[1;31mp\033[0m"
#define PGREEN "\033[1;32mp\033[0m"
#define PYELLOW "\033[1;33mp\033[0m"
#define PBLUE "\033[1;34mp\033[0m"
#define PMAGENTA "\033[1;35mp\033[0m"
#define PCYAN "\033[1;36mp\033[0m"
#define PWHITE "\033[1;37mp\033[0m"
  while (1) {
    pthread_mutex_lock(&mutex);
    while ((count) == (buf_size)) {
      printf(PRED);
      fflush(stdout);
      pthread_cond_wait(&modify, &mutex);
    }
    append(rand() % 10);
    printf(PYELLOW);
    fflush(stdout);
    pthread_cond_signal(&modify);
    pthread_mutex_unlock(&mutex);
  }
}
void *consumer(void *arg) {
#define RED "\033[1;31mc%01d\033[0m"
#define GREEN "\033[1;32mc%01d\033[0m"
#define YELLOW "\033[1;33mc%01d\033[0m"
#define BLUE "\033[1;34mc%01d\033[0m"
#define MAGENTA "\033[1;35mc%01d\033[0m"
#define CYAN "\033[1;36mc%01d\033[0m"
#define WHITE "\033[1;37mc%01d\033[0m"
  auto id = *(static_cast<uint32_t *>(arg));
  while (1) {
    pthread_mutex_lock(&mutex);
    while ((0) == (count)) {
      printf(RED, id);
      fflush(stdout);
      pthread_cond_wait(&modify, &mutex);
    }
  }
  head();
  printf(YELLOW, id);
  fflush(stdout);
  pthread_cond_signal(&modify);
  pthread_mutex_unlock(&mutex);
}
int main(int argc, char **argv) {
  if ((argc) < (4)) {
    printf("Usage: ./main <buffer_size> <#producers> <#consumers>\n");
    exit(1);
  }
  g_start_time = std::chrono::high_resolution_clock::now();
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ", "start",
          " ", " argc='", fmt::format("{}", argc), "'", " argv[0]='",
          fmt::format("{}", argv[0]), "'"});
  srand(999);
  buf_size = atoi(argv[1]);
  numProducers = atoi(argv[2]);
  numConsumers = atoi(argv[3]);
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ",
          "initiate mutex and condition variable", " "});
  pthread_mutex_init(&mutex, nullptr);
  pthread_cond_init(&modify, nullptr);
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ",
          "allocate buffer", " ", " buf_size='", fmt::format("{}", buf_size),
          "'"});
  buffer = static_cast<uint32_t *>(malloc(((buf_size) * (sizeof(uint32_t)))));
  pthread_t prods[numProducers], cons[numConsumers];
  uint32_t threadIds[numConsumers], i;
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ",
          "start consumers", " ", " numConsumers='",
          fmt::format("{}", numConsumers), "'"});
  for (auto i = 0; (i) < (numConsumers); (i) += (1)) {
    threadIds[i] = i;
    pthread_create(((cons) + (i)), nullptr, consumer, ((threadIds) + (i)));
  }
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ",
          "start producers", " ", " numProducers='",
          fmt::format("{}", numProducers), "'"});
  for (auto i = 0; (i) < (numProducers); (i) += (1)) {
    pthread_create(((prods) + (i)), nullptr, producer, nullptr);
  }
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ",
          "wait for threads to finish", " "});
  for (auto i = 0; (i) < (numProducers); (i) += (1)) {
    pthread_join(prods[i], nullptr);
  }
  for (auto i = 0; (i) < (numConsumers); (i) += (1)) {
    pthread_join(cons[i], nullptr);
  }
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ",
          "leave program", " "});
  return 0;
}