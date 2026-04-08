(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))




(progn
  (defparameter *source-dir* #P"example/196_forthjit/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname 'cl-cpp-generator2 *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (defun lprint (&key (msg "")
		   (vars nil)
		   )
    #+nil `(<< std--cout
	       (std--format
		(string ,(format nil "~a~{ ~a='{}'~}\\n"
				 msg
				 (loop for e in vars collect (emit-c :code e  :omit-redundant-parentheses t)) ))
		,@vars))
    #-more
    ""
    #+more
    `(<< std--cout
	 (string ,(format nil "~a"
			  msg
			  
			  ))
	 ,@(loop for e in vars
		 appending
		 `((string ,(format nil " ~a='" (emit-c :code e :omit-redundant-parentheses t)))
		   ,e
		   (string "' ")))   
	 std--endl))
  
  (write-source
   (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames "main.cpp" *source-dir*))
   `(do0
     (include<> iostream
		vector
		string
		unordered_map
		sstream
		algorithm
		libgccjit++.h)
     "using namespace gccjit;"
     (space enum class Error (curly (comma Unknown_Word Stack_Error Compile_Error)))
     (defclass+ ForthVM ()
	 "static constexpr auto MAX_STACK =  256;"
       "static constexpr auto MAX_DICT =  64;"
       "static constexpr auto FUEL_LIMIT =  10'000;"
       "std::vector<int> stack;"
       "std::unordered_map<std::string, int> variables;"
       "std::unordered_map<std::string, void(*)()> dictionary;"
       "int fuel = 0;"
       "public:"
       (defmethod push (val)
	 (declare (type int val)
		  (values void))
	 (when (<= MAX_STACK (stack.size))
	   (throw Error--Stack_Error))
	 (stack.push_back val))

       (defmethod pop ()
	 (declare (values int))
	 (when (stack.empty)
	   (throw Error--Stack_Error))
	 (let ((val (stack.back)))
	   (stack.pop_back)
	   (return val)))

       (defmethod consume_fuel ()
	 (when (< FUEL_LIMIT
		  "++fuel")
	   (throw Error--Stack_Error)))

       (defmethod dot ()
	 (<< std--cout (pop) (string " ")))

       (defmethod dup ()
	 (let ((v (pop)))
	   (push v)
	   (push v)))

       (defmethod drop ()
	 (pop))

       (defmethod swap ()
	 (let ((a (pop))
	       (b (pop)))
	   (push a)
	   (push b))))

     (defclass+ JITCompiler ()
       "context ctx;"
       "ForthVM& vm;"
       "public:"
       (defmethod JITCompiler (v)
	 (declare (type ForthVM& v)
		  (explicit)
		  (constructs (ctx (context--acquire))
			      (vm v))))
       (defmethod compile_word (name tokens)
	 (let ((result (ctx.compile))))))

     (defun to_upper (s)
       (declare (type "std::string" s)
		(values "std::string"))
       (std--transform (s.begin)
		       (s.end)
		       (s.begin)
		       --toupper)
       (return s))

     (defun interpreter_loop ()
       (let ((vm "ForthVM{}")
	     (input "std::string{}"))))
     
     
     (defun main (argc argv)
       (declare (type int argc) (type char** argv) (values int))
       (let ((ctx (gccjit--context--acquire))
	     (int_type (ctx.get_type GCC_JIT_TYPE_INT))
	     (param_i (ctx.new_param int_type
				     (string "i")))
	     )
	 "std::vector<gccjit::param> params = { param_i };"
	 (let ((func (ctx.new_function
		      GCC_JIT_FUNCTION_EXPORTED
		      int_type
		      (string "square")
		      params
		      0))
	       (block (func.new_block (string "entry")))
	       (i_rval param_i))
	   (block.end_with_return (* i_rval i_rval))
	   (let ((*result (ctx.compile)))
	     (unless result
	       ,(lprint :msg "compilation failed")
	       (return 1)))
	   (let ((square ("reinterpret_cast<int (*)(int)>" (gcc_jit_result_get_code
							    result (string "square"))))))
	   (let ((val 5)
		 (sq (square val)))
	     ,(lprint :msg "result"
		      :vars `(val sq)))
	   (gcc_jit_result_release result)))
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
