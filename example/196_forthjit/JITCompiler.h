#pragma once
#include <functional>
#include <libgccjit++.h>
#include <string>
#include <variant>
#include <vector>
class ForthVM;
using CompiledWord = int (*)(ForthVM *);
class JITCompiler {
