#pragma once
// header 
#include <libgccjit++.h>
#include <string>
#include <vector>
#include <functional> 
#include "Operation.h" 
class ForthVM; 
using CompiledWord = int (*)(ForthVM *); 
class JITCompiler  {
        public:
        struct Result {
                gcc_jit_result *jit_result{nullptr}; 
                CompiledWord function{nullptr}; 
};
        JITCompiler::Result compile_word (const std::string& symbol_name, const std::vector<Operation>& operations)       ;   
};
