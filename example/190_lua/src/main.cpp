#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
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

static int panic_handler(lua_State *L) {
  auto *msg{lua_tostring(L, -1)};
  std::cout << "" << " msg='" << msg << "' " << std::endl;
  while (true) {
  }
  return 0;
}

static int native_add(lua_State *L) {
  auto a{luaL_checknumber(L, 1)};
  auto b{luaL_checknumber(L, 2)};
  lua_pushnumber(L, a + b);
  return 1;
}

int main() {
  auto *L{lua_newstate(custom_alloc, nullptr)};
  if (nullptr == L) {
    std::cout << "Failed to init Lua state (OOM)" << std::endl;
    return -1;
  }
  lua_atpanic(L, panic_handler);
  // lade nur was noetig ist, kein luaL_openlibs
  luaL_requiref(L, "_G", luaopen_base, 1);
  lua_pop(L, 1);
  lua_pushcfunction(L, native_add);
  lua_setglobal(L, "cpp_add");
  auto *lua_script{R"(local x = 10
local y = 20
local sum = cpp_add(x,y)
print('Sum: ' .. sum)
lua_status = 'OK')"};
  auto status{luaL_dostring(L, lua_script)};
  if (LUA_OK == status) {
    lua_getglobal(L, "lua_status");
    if (lua_isstring(L, -1)) {
      auto stat{lua_tostring(L, -1)};
      std::cout << "" << " stat='" << stat << "' " << std::endl;
    }
    lua_pop(L, 1);
  } else {
    if (lua_isstring(L, -1)) {
      auto err{lua_tostring(L, -1)};
      std::cout << "" << " err='" << err << "' " << std::endl;
    }
    lua_pop(L, 1);
  }
  lua_close(L);
  return 0;
}
