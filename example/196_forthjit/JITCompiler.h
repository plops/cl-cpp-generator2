#pragma once
// header 
class JITCompiler  {
        public:
        struct Result prgon(gcc_jit_result *jit_result{nullptr};, CompiledWord function{nullptr};);
        REsult compile_word (const std::string& symbol_name, const std::vector<Operation>& operations)       ;   
};
