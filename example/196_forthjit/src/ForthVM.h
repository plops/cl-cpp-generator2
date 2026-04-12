#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "JITCompiler.h"
#include "Operation.h"

class ForthVM {
    static constexpr auto MAX_STACK  = 256;
    static constexpr auto MAX_DICT   = 64;
    static constexpr auto FUEL_LIMIT = 10'000;

public:
    struct VariableEntry {
        std::string name;
        int         value{0};
    };
    ;

    struct WordEntry {
        std::string     name;
        gcc_jit_result *jit_result;
        CompiledWord    function;
    };
    ;

    bool                                 compile_mode_{false};
    int                                  fuel_{1000};
    std::vector<int>                     data_stack_;
    std::string                          pending_name_;
    std::vector<std::string>             pending_tokens_;
    std::vector<VariableEntry>           variables_;
    std::vector<WordEntry>               words_;
    std::unordered_map<std::string, int> variable_lookup_;
    std::unordered_map<std::string, int> word_lookup_;

    ForthVM();

    ~ForthVM();

    void execute_line(const std::string &line);

    void abort_pending_definition();

    int push_literal(int value);

    int add();

    int sub();

    int mul();

    int dup();

    int drop();

    int swap();

    int dot();

    int lessthan();

    int greaterthan();

    int equal();

    int fetch();

    int store();

    int call_word(int index);

    int pop_condition(int *out);

protected:
    int consume_fuel();

    int push_raw(int value);

    void begin_definition(const std::string &name);

    void define_variable(const std::string &name);

    std::size_t consume_definition_tokens(const std::vector<std::string> &tokens, std::size_t start_index);

    static std::vector<Operation> parse_operations(const std::vector<std::string> &tokens, int mode);

    void finish_definition();

    void execute_segment(const std::vector<std::string> &tokens);

    bool is_dictionary_full();
};
