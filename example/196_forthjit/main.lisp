(in-package :cl-cpp-generator2)

(write-source
 (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames "main.cpp" *source-dir*))
 `(do0
   (include<> algorithm
	      array
	      charconv
	      cctype
	      cstdlib
	      functional
	      iostream
	      libgccjit++.h
	      memory
	      optional
	      sstream
	      string
	      string_view
	      unordered_map
	      utility
	      vector)
   (include "helpers.h")
   "using namespace gccjit;"
   (space namespace
	  (progn

	    "class ForthVM;"
	    "using CompiledWord = int (*)(ForthVM*);"

	    (defun normalize_dictionary_name (text)
	      (declare (type "std::string_view" text)
		       (values "std::string"))
	      "constexpr std::size_t kMaxNameLength{31};"
	      (let ((normalized (to_upper text)))
		(when (< kMaxNameLength (normalized.size))
		  (normalized.resize kMaxNameLength))
		(return normalized)))

	    

	    (defun parse_integer (token)
	      (declare (type "std::string_view" token)
		       (values "std::optional<int>"))
	      (when (token.empty)
		(return std--nullopt))
	      (let ((value 0)
		    (*begin (token.data))
		    (*end (+ (token.data)
			     (token.size)))
		    ((bracket ptr ec)
		      (std--from_chars begin end value)))
		(when (logior (!= ec "std::errc{}")
			      (!= ptr end))
		  (return std--nullopt))
		(return value)))

	    (defun error_name (error)
	      (declare (type "Error" error)
		       (values "const char*"))
	      (case error
		,@(loop for e in `(Unknown_Word Stack_Error Compile_Error)
			collect
			`(,(format nil "Error::~a" e)
			  (return (string ,e )))))
	      (return (string "Compile_error")))

	    

	    ,@(loop for e in `(add sub mul)
		    collect
		    `(defun ,(format nil "checked_~a" e)
			 (lhs rhs result)
		       (declare (type int lhs rhs)
				(type int* result)
				(values bool))
		       (return (,(format nil "!__builtin_~a_overflow" e)
				lhs rhs result))))

	    (defun lookup_primitive (token)
	      (declare (type std--string_view token)
		       (values "std::optional<Primitive>"))
	      (let ((upper (to_upper token)))
		,@(loop for e in l-prim
			collect
			(destructuring-bind (&key name symbol short)
			    e
			  `(when (== (string ,symbol)
				     upper)
			     (return ,(format nil "Primitive::~a" name))))
			)
		(return std--nullopt)))

	    (defun is_reserved_token (token)
	      (declare (type "std::string_view" token)
		       (values bool)
		       )
	      (let ((upper (to_upper token)))
		(return (logior ,@(loop for e in `(IF ELSE THEN VARIABLE ":" ";")
					collect
					`(== (string ,(format nil "~a" e))
					     upper))))))

	    ))
   
   
   

   (defun interpreter_loop ()
     (let ((vm "ForthVM{}")
	   (input "std::string{}"))
       (while (std--getline std--cin input)
	      (let ((ss (std--stringstream input))
		    (token "std::string{}")))
	      (handler-case
		  (while (>> ss token)
			 (vm.consume_fuel)
			 (let ((cmd (to_upper token)))))))))
   
   
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
 :tidy t)
