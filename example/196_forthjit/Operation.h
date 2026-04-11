#pragma once
// header 
#include "helpers.h" 
#include <vector> 
class Operation  {
        public:
         Operation ()       ;   
         ~Operation ()       ;   
        Operation literal (int value)       ;   
        private:
        OperationKind kind {OperationKind::Literal};
        int value {0};
        Primitive primitive {Primitive::Add};
        std::vector<Operation> true_branch;
        std::vector<Operation> false_branch;
};
