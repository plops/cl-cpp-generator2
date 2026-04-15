#pragma once
// header 
#include "helpers.h" 
#include <vector> 
class Operation  {
        public:
         Operation ()       ;   
         ~Operation ()       ;   
        static Operation literal (int value)       ;   
        static Operation primitive_op (Primitive primitive)       ;   
        static Operation call_word (int word_index)       ;   
        static Operation if_op (std::vector<Operation> true_branch, std::vector<Operation> false_branch)       ;   
        OperationKind kind {OperationKind::Literal};
        int value {0};
        Primitive primitive {Primitive::Add};
        std::vector<Operation> true_branch;
        std::vector<Operation> false_branch;
};
