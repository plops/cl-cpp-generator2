#ifndef OPERATION_H
#define OPERATION_H

// header 
class Operation  {
        public:
         Operation ()       ;   
         ~Operation ()       ;   
        private:
        OperationKind kind {OperationKind::Literal};
        int value {0};
        Primitive primitive {Primitive::Add};
        std::vector<Operation> true_branch;
        std::vector<Operation> false_branch;
};

#endif /* !OPERATION_H */