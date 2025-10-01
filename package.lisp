(defpackage :cl-cpp-generator2  
  (:use :cl :alexandria)  
  (:export  
    ;; Core API  
    :write-source :emit-c :write-notebook  
  
    ;; --- punctuation / layout ---  
    :comma :semicolon :space :space-n :indent :comments :lines :doc  
  
    ;; --- grouping / bracketing / calls ---  
    :paren :paren* :angle :bracket :curly :designated-initializer :scope :split-header-and-code  
  
    ;; --- call / member access / call variants ---  
    :dot :->  
  
    ;; --- structural / composite forms ---  
    :do :do0 :progn :namespace  
  
    ;; --- preprocessor / include / file helpers ---  
    :pragma :include :include<>  
  
    ;; --- declarations / type & cast helpers ---  
    :cast :new :deftype :typedef  ; (typedef kept for convenience if you use it)  
    :let :letc :letd :setf :using  
  
    ;; --- definitions: functions / methods / lambdas ---  
    :lambda  
    :defun :defun* :defun+  
    :defmethod  
    :return :co_return :co_await :co_yield :throw  
  
    ;; --- classes / structs / access specifiers ---  
    :defclass :defclass+ :protected :public :struct :defstruct0  
  
    ;; --- control flow / selection ---  
    :if :if-constexpr :when :unless :cond :case  
  
    ;; --- looping / iteration ---  
    :for :for-range :foreach :dotimes :while  
  
    ;; --- array / container access / slicing ---  
    :aref :slice  
  
    ;; --- unary / reference helpers ---  
    :not :bitwise-not :deref :ref  
  
    ;; --- arithmetic / numeric ---  
    :+ :- :* :/ :% :<< :>> :^ :xor  
  
    ;; --- bitwise / boolean operators ---  
    :& :logand :logior ; :logxor if you add it later  
  
    ;; --- boolean combinators (logical / short-circuit) ---  
    :and :or  
  
    ;; --- comparison / relational ---  
    :< :<= :> :>= :== :!=  
  
    ;; --- assignment / compound assignment / update ---  
    ;:= ; note: use setf for assignment (except maybe in for loops)
    
    :incf :decf  
    :*= :/= :^= :/= ; (compound ops present in emit-c: *=, /=, ^=)  
    
  
    ;; --- shifts / modulus ---  
    :<< :>>  
  
    ;; --- strings / chars / literals ---  
    :string :string-r :string-u8 :char :hex :?  
  
    ;; --- misc / error handling ---  
    :handler-case :throw  
  
    ;; --- debugging / internal helpers (optional) ---  
    :scope :lines :doc  
  ))  


