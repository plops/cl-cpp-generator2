#pragma once
#include <concepts>
#include <cstring>
#include <type_traits>
template <typename T> constexpr inline T *c_resource_null_value = nullptr;
// two api schemas for destructors and constructor
// 1) thing* construct();      void destruct(thing*)
// 2) void construct(thing**); void destruct(thing**)
//
// modifiers like replace(..) exist only for schema 2) act like auto
// construct(thing**)
template <typename T, auto *ConstructFunction, auto *DestructFunction>
class c_resource {
  using pointer = T *;
  using const_pointer = std::add_const_t<T> *;
};
