#include <array>
#include <cstdio>
#include <vector>
// check luaconf.h, LUA_32BITS LUA_NUMBER=float LUA_INT_TYPE=long LUA_IDSIZE
// reduce debug info to save memory
extern "C" {
#include "lauxlib.h"
#include "lua.h"
#include "lualib.h"
};
constexpr size_t LUA_HEAP_SIZE = 16 * 1024;
static std::array<uint8_t, LUA_HEAP_SIZE> lua_memory_pool;
static size_t lua_mem_used = 0;

static void *custom_alloc(void *ud, void *ptr, size_t osize, size_t nsize) {
  if (0 == nsize) {
    return nullptr;
  }
  if (LUA_HEAP_SIZE < lua_mem_used + nsize) {
    return nullptr;
  }
  void *new_ptr = &lua_memory_pool[lua_mem_used];
  lua_mem_used += nsize;
  return new_ptr;
}

int main() { return 0; }
