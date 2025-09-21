#ifndef MYCLASS_H
#define MYCLASS_H

// header 
class MyClass  {
        public:
         MyClass ()       ;   
         ~MyClass ()       ;   
        
        const int& getValue ()       ;   
        void setValue (int value)       ;   
        private:
        int value {0};
};

#endif /* !MYCLASS_H */