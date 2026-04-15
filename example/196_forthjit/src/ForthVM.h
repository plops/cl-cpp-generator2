#pragma once
#include <vector>
#include <string>
#include <unordered_map> 
#include "Operation.h" 
#include "JITCompiler.h" 

int to_status (Error error)      ;  

std::string to_upper (std::string_view text)      ;  

std::vector<std::string> split_on_spaces (const std::string& line)      ;  
class ForthVM  {
        static constexpr auto MAX_STACK =  256; 
        static constexpr auto MAX_DICT =  64; 
        static constexpr auto FUEL_LIMIT =  10'000; 
        public:
            struct VariableEntry {
                std::string name; 
                int value{0}; 
}; ;
            struct WordEntry {
                std::string name; 
                gcc_jit_result* jit_result; 
                CompiledWord function; 
}; ;
        bool compile_mode_ {false};
        int fuel_ {1000};
        std::vector<int> data_stack_;
        std::string pending_name_;
        std::vector<std::string> pending_tokens_;
        std::vector<VariableEntry> variables_;
        std::vector<WordEntry> words_;
        std::unordered_map<std::string, int> variable_lookup_;
        std::unordered_map<std::string, int> word_lookup_;
         ForthVM ()       ;   
         ~ForthVM ()       ;   
        void process_line (const std::string& line)       ;   
        void abort_pending_definition ()       ;   
        int push_literal (int value)       ;   
        int add ()       ;   
        int sub ()       ;   
        int mul ()       ;   
        int dup ()       ;   
        int drop ()       ;   
        int swap ()       ;   
        int dot ()       ;   
        int lessthan ()       ;   
        int greaterthan ()       ;   
        int equal ()       ;   
        int fetch ()       ;   
        int store ()       ;   
        int call_word (int index)       ;   
        int pop_condition (int* out)       ;   
        protected:
        int consume_fuel ()       ;   
        int push_raw (int value)       ;   
        void begin_definition (const std::string& name)       ;   
        void define_variable (const std::string& name)       ;   
        std::size_t consume_definition_tokens (const std::vector<std::string>& tokens, std::size_t start_index)       ;   
        std::vector<Operation> parse_operations (const std::vector<std::string>& tokens, ParseMode mode) const      ;   
        ParseResult parse_sequence (const std::vector<std::string>& tokens, std::size_t index, ParseMode mode) const      ;   
        Operation resolve_token (const std::string& token, ParseMode mode) const      ;   
        int execute_operations (const std::vector<Operation>& operations)       ;   
        int execute_operation (const Operation& operation)       ;   
        int execute_primitive (Primitive primitive)       ;   
        void validate_definition_line (const std::vector<std::string>& tokens) const      ;   
        bool is_reserved_name (const std::string& name) const      ;   
        void finish_definition ()       ;   
        void execute_segment (const std::vector<std::string>& tokens)       ;   
        bool is_dictionary_full ()       ;   
};

int forth_push_literal (ForthVM* vm, int value)      ;  

int forth_add (ForthVM* vm)      ;  

int forth_sub (ForthVM* vm)      ;  

int forth_mul (ForthVM* vm)      ;  

int forth_dup (ForthVM* vm)      ;  

int forth_drop (ForthVM* vm)      ;  

int forth_swap (ForthVM* vm)      ;  

int forth_dot (ForthVM* vm)      ;  

int forth_lt (ForthVM* vm)      ;  

int forth_gt (ForthVM* vm)      ;  

int forth_eq (ForthVM* vm)      ;  

int forth_fetch (ForthVM* vm)      ;  

int forth_store (ForthVM* vm)      ;  

int forth_pop_condition (ForthVM* vm, int* out_condition)      ;  

int forth_call_word (ForthVM* vm, int word_index)      ;  
