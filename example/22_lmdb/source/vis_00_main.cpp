
#include "utils.h"

#include "globals.h"

;
// g++ -march=native -Ofast --std=gnu++20 vis_00_main.cpp
// -I/media/sdb4/cuda/11.0.1/include/ -L /media/sdb4/cuda/11.0.1/lib -lcudart
// -lcuda
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

#include "lmdbpp/lmdb++.h"

using namespace std::chrono_literals;
State state = {};
int main(int argc, char const *const *const argv) {
  state._main_version = "e8faa6ee8fe1a3b8b367708dadf9436c4309ae17";
  state._code_repository =
      "https://github.com/plops/cl-cpp-generator2/tree/master/example/19_nvrtc";
  state._code_generation_time = "09:15:38 of Sunday, 2020-06-28 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start main") << (" ")
      << (std::setw(8)) << (" state._main_version='") << (state._main_version)
      << ("'") << (std::setw(8)) << (" state._code_repository='")
      << (state._code_repository) << ("'") << (std::setw(8))
      << (" state._code_generation_time='") << (state._code_generation_time)
      << ("'") << (std::endl) << (std::flush);
  try {
    auto env = lmdb::env::create();
    env.set_mapsize(((1UL) * (1024UL) * (1024UL) * (1024UL)));
    env.open("./example.mdb");
    auto wtxn = lmdb::txn::begin(env);
    auto dbi = lmdb::dbi::open(wtxn, nullptr);
    dbi.put(wtxn, "username", "jhacker");
    dbi.put(wtxn, "email", "jhacker@example.org");
    dbi.put(wtxn, "fullname", "J. Random Hacker");
    wtxn.commit();
    auto rtxn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
    auto cursor = lmdb::cursor::open(rtxn, dbi);
    auto key = std::string();
    auto value = std::string();
    while (cursor.get(key, value, MDB_NEXT)) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("") << (" ") << (std::setw(8)) << (" key='")
                  << (key) << ("'") << (std::setw(8)) << (" value='") << (value)
                  << ("'") << (std::endl) << (std::flush);
    }
    cursor.close();
    rtxn.abort();
  } catch (const std::exception &e) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("error") << (" ") << (std::setw(8)) << (" e.what()='")
                << (e.what()) << ("'") << (std::endl) << (std::flush);
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};