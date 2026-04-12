(in-package :cl-cpp-generator2)

(let* ((class-name 'JITCompiler))
  (write-class
   :dir *full-source-dir*
   :name class-name
   :headers '()
   :header-preamble `(do0 (include<> libgccjit++.h string vector functional variant)
			  "class ForthVM;"
			  "using CompiledWord = int (*)(ForthVM *);")
   :implementation-preamble `(do0 (include "Operation.h")
				  (include "helpers.h"))
   :code `(do0
	   (defclass ,class-name ()
	     "public:"
	     (space struct Result (progn
				    "gcc_jit_result *jit_result{nullptr};"
				    "CompiledWord function{nullptr};"))

	     (defmethod compile_word (symbol-name operations)
	       (declare (type "const std::string&" symbol-name)
			(type "const std::vector<Operation>&" operations)
			(values Result))
	       (let ((ctx (gccjit--context--acquire))
		     (int-type (ctx.get_type GCC_JIT_TYPE_INT))
		     (vm-struct (ctx.new_opaque_struct_type (string "ForthVM")))
		     (vm-ptr-type (vm-struct.get_pointer))
		     (param-vm (ctx.new_param vm-ptr-type (string "vm")))
		     (word-params (space std--vector (angle gccjit--param) (curly param-vm)))
		     (function (ctx.new_function GCC_JIT_FUNCTION_EXPORTED
						 int-type
						 symbol-name
						 word-params
						 0))
                     (entry-block (function.new_block (string "entry"))))
                 (entry-block.end_with_return (ctx.zero int-type))
		 (let ((jit-result (ctx.compile)))
		   (unless jit-result
		     (throw Error--Compile_Error))
		   (let ((symbol (gcc_jit_result_get_code jit-result (dot symbol-name (c_str)))))
		     (unless symbol
		       (gcc_jit_result_release jit-result)
		       (throw Error--Compile_Error))
		     (return (curly (= .jit_result jit-result)
				    (= .function (reinterpret_cast CompiledWord symbol))))))))))
   :format t))
