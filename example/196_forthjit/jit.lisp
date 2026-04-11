(in-package :cl-cpp-generator2)

(let* ((class-name `JITCompiler)
       (members0 `((:name ctx :type  context  :initform 0 ;(context--acquire)
			  )
		   (:name vm :type  ForthVM&  :initform 0)))
       (members (loop for e in members0
		      collect
		      (destructuring-bind (&key name type param doc initform) e
			`(:name ,name
			  :type ,type
			  :param ,param
			  :doc ,doc
			  :initform ,initform
			  :member-name ,(intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))
			  :param-name ,(when param
					 (intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))))))))
     
  (write-class
   :dir *full-source-dir*
   :name class-name
   :headers `()
   :header-preamble `(do0 (comments "header"))
   :implementation-preamble `(do0 (comments "implementation"))
   :code `(do0
	   (defclass ,class-name ()
	     "public:"
	     (space struct Result (prgon
				   "gcc_jit_result *jit_result{nullptr};"
				   "CompiledWord function{nullptr};"))

		
	     (defmethod compile_word (symbol_name operations)
	       (declare (type "const std::string&" symbol_name)
			(type "const std::vector<Operation>&" operations)
			(values REsult))
	       (let ((ctx (gccjit--context--acquire))
		     (int_type (ctx.get_type GCC_JIT_TYPE_INT))
		     (vm_struct (ctx.new_opaque_struct_type (string "ForthVM")))
		     (vm_ptr_type (vmstruct.get_pointer))
		     (int_ptr_type (int_type.get_pointer))
		     (param_vm (ctx.new_param vm_ptr_type (string "vm")))
		     (word_params param_vm)
		     (function (ctx.new_function GCC_JIT_FUNCTION_EXPORTED
						 int_type
						 symbol_name
						 word_params
						 0))
		     (declare_helper (lambda (name params)
				       (declare (type "const std::string&" name)
						(type "std::vector<param>" params))
				       (return (ctx.new_function GCC_JIT_FUNCTION_IMPORTED
								 int_type
								 name
								 params
								 0))))
		     (make_vm_only_helper (lambda (name)
					    (declare (type "const std::string&" name))
					    (let ((helper_vm (ctx.new_param vm_ptr_type (string "vm")))
						  (params (curly helper_vm )))
					      (declare (type "std::vector<params>" params))
					      (return (declare_helper name params)))))
		     (make_vm_int_helper (lambda (name)
					   (declare (type "const std::string&" name))
					   (let ((helper_vm (ctx.new_param vm_ptr_type (string "vm")))
						 (helper_value (ctx.new_param int_type (string "value")))
						 (params (curly helper_vm helper_value)))
					     (declare (type "std::vector<params>" params))
					     (return (declare_helper name params)))))
		     ,@(loop for e in `(add sub mul dup drop swap dot lt gt eq fetch store)
			     collect
			     `(,(format nil "helper_~a" e)
			       (make_vm_only_helper (string ,(format nil "forth_~a" e)))))
		     ,@(loop for e in `(push_literal call_word)
			     collect
			     `(,(format nil "helper_~a" e)
			       (make_vm_int_helper (string ,(format nil "forth_~a" e)))))
		     (pop_vm (ctx.new_param vm_ptr_type (string "vm")))
		     (pop_out (ctx.new_param int_ptr_type (string "out_condition")))
		     (pop_params (curly pop_vm pop_out))
		     (helper_pop_condition (declare_helper (string "forth_pop_condition")
							   pop_params))
		     (entry_block (function.new_block (string "entry")))
		     (error_block (function.new_block (string "error")))
		     (error_value (function.new_local int_type (string "error_value"))))
		 (declare (type "std::vector<param>" word_params pop_params))
		 (entry_block.add_assignment error_value (ctx.zero int_type))
		 (let ((block_counter 0)
		       (fresh_block_name (lambda (prefix)
					   (declare (type "std::string_view" prefix)
					;(values "std::string")
						    )
					   (let ((res (space std--string (curly prefix))))
					     (incf res (string "_"))
					     (incf res (std--to_string block_counter++)))
					   (return res)))
		       (emit_checked_call (lambda (current_block helper args)
					    (declare (type block current_block)
						     (type "gccjit::function" helper)
						     (type "const std::vector<rvalue>&" args)
						     (values block))
					    (let ((ok_block (function.new_block (fresh_block_name (string "ok"))))
						  (mutable_args args))
					      (current_block.add_assignment error_value
									    (ctx.new_call helper mutable_args))
					      (current_block.end_with_conditional (ctx.new_eq error_value
											      (ctx.zero int_type))
										  ok_block
										  error_block))))
		       (emit_operations (lambda (current_block
						 ops)
					  (declare (type block current_block)
						   (type "const std::vector<Operation>&" ops)
						   (values block))
					  (for-range (operation ops)
						     (declare (type "const auto &" operation))
						     (case operation.kind
						       (OperationKind--Literal
							(setf current_block (emit_checked_call current_block helper_push_literal
											       (curly param_vm
												      (ctx.new_rvalue int_type
														      operation.value)))))
						       (OperationKind--Primitive
							(let ((helper helper_add))
							  (case operation.primitive
							    ,@(loop for e in *l-prim*
								    collect
								    (destructuring-bind (&key name symbol short) e
								      `(,(format nil "Primitive::~a" name)
									(setf helper ,(format nil "helper_~a" short)))))))
							(setf current_block (emit_checked_call current_block helper (curly param_vm))))
						       (OperationKind--CallWord
							(setf current_block (emit_checked_call current_block helper_call_word
											       (curly param_vm
												      (ctx.new_rvalue int_type
														      operation.value)))))
						       (OperationKind--If
							(let ((condition_value
								(function.new_local int_type
										    (fresh_block_name (string "condition"))))
							      )
							  (setf current_block (emit_checked_call current_block
												 helper_pop_condition
												 (curly param_vm
													(condition_value.get_address))))
							  (let ((true_block (function.new_block (fresh_block_name (string "if_true"))))
								(false_block (function.new_block (fresh_block_name (string "if_false"))))
								(after_block (function.new_block (fresh_block_name (string "after_if")))))
							    (current_block.end_with_conditional (ctx.new_ne condition_value
													    (ctx.zero int_type)
													    true_block
													    false_block))
							    (let ((completed_true (emit_operations true_block
												   operation.true_branch)))
							      (completed_true.end_with_jump after_block)
							      (let ((completed_false (emit_operations false_block operations.false_branch)))
								(compleded_false.end_with_jump after_block)
								(setf current_block after_block)))))))
						     )
					  (return current_block)))
		       (completed_entry (emit_operations entry_block operations)
					)
			  
			  
			  
		       )
		      
		   )
		 (completed_entry.end_with_return (ctx.zero int_type))
		 (error_block.end_with_return error_value)
		 (let ((*jit_result (ctx.compile)))
		   (unless jit_result
		     (throw Error--Compile_Error))
		   (let ((*symbol (gcc_jit_result_get_code jit_result (symbol_name.c_str))))
		     (unless symbol
		       (gcc_jit_result_release jit_result)
		       (throw Error--Compile_Error))
		     (return (curly (= .jit_result jit_result)
				    (= .function ("reinterpret_cast<CompiledWord>" symbol)))))))

	       )
	     ))
   :format t
   ))
