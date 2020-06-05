
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
comment(Embedding Lua in C++ #1  https://www.youtube.com/watch?v=4l5HdmPoynw);
extern "C" {
#include "lua/lauxlib.h"
#include "lua/lua.h"
#include "lua/lualib.h"
};
bool checkLua (lua_State* L, int res){
  if (!((res) == (LUA_OK))) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("lua_error") << (" ") << (std::setw(8)) << (" res='")
                << (res) << ("'") << (std::setw(8))
                << (" lua_tostring(L, -1)='") << (lua_tostring(L, -1)) << ("'")
                << (std::endl) << (std::flush);
    return false;
  };
  return true;
}
int lua_HostFunction (lua_State* L){
  auto a = static_cast<float>(lua_tonumber(L, 1));
  auto b = static_cast<float>(lua_tonumber(L, 2));

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("HostFunction") << (" ")
      << (std::setw(8)) << (" a='") << (a) << ("'") << (std::setw(8))
      << (" b='") << (b) << ("'") << (std::endl) << (std::flush);
  auto c = ((a) * (b));
  lua_pushnumber(L, c);
  return 1;
}
void initLua (){
  state._lua_state = luaL_newstate();
  std::string cmd = "a = 7+11+math.sin(23.7)";
  lua_State *L = state._lua_state;
  luaL_openlibs(L);
  lua_register(L, "HostFunction", lua_HostFunction);
  auto res = luaL_dofile(L, "init.lua");
  if ((res) == (LUA_OK)) {
    lua_getglobal(L, "a");
    if (lua_isnumber(L, -1)) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("lua_ok") << (" ") << (std::setw(8))
                  << (" static_cast<float>(lua_tonumber(L, -1))='")
                  << (static_cast<float>(lua_tonumber(L, -1))) << ("'")
                  << (std::endl) << (std::flush);
    };
    lua_getglobal(L, "DoAThing");
    if (lua_isfunction(L, -1)) {
      lua_pushnumber(L, (3.50f));
      lua_pushnumber(L, (7.10f));
      if (checkLua(L, lua_pcall(L, 2, 1, 0))) {

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("lua") << (" ") << (std::setw(8))
                    << (" static_cast<float>(lua_tonumber(L, -1))='")
                    << (static_cast<float>(lua_tonumber(L, -1))) << ("'")
                    << (std::endl) << (std::flush);
      };
    };
  } else {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("lua_error") << (" ") << (std::setw(8)) << (" res='")
                << (res) << ("'") << (std::setw(8))
                << (" lua_tostring(L, -1)='") << (lua_tostring(L, -1)) << ("'")
                << (std::endl) << (std::flush);
  };
}
void cleanupLua (){
  lua_close(state._lua_state);
};