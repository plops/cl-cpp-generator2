(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")c
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(progn
  (defparameter *source-dir* #P"example/196_forthjit/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname 'cl-cpp-generator2 *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  
  (load "util.lisp")
  
  


  (let* ((class-name `Operation)
	  (members0 `((:name kind   :type OperationKind :initform OperationKind--Literal)
		      (:name value :type  int  :initform 0)
		      (:name primitive   :type Primitive :initform Primitive--Add)
		      (:name true_branch :type "std::vector<Operation>")
		      (:name false_branch :type "std::vector<Operation>")))
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
      :header-preamble `(do0 (comments "header") (include "helpers.h") (include<> vector))
      :implementation-preamble `(do0 (comments "implementation"))
      :code `(do0
	      (defclass ,class-name ()
		"public:"
		
		(defmethod ,class-name (&key ,@(remove-if
						#'null
						(loop for e in members
						      collect
						      (destructuring-bind (&key name type param doc initform param-name member-name) e
							(when param
							  `(,param-name ,(if initform initform 0)))))))
		  (declare
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (&key name type param doc initform param-name member-name) e
					(let ((const-p (let* ((s  (format nil "~a" type))
							      (last-char (aref s (- (length s)
										    1))))
							 (not (eq #\* last-char)))))
					  (when param
					    (if (eq name 'callback)
						`(type "std::function<void(const uint8_t*, const size_t)>"
						       #+nil PacketReceivedCallback ,param-name)
						`(type ,(if const-p
							    (format nil "const ~a&" type)
							    type)
						       ,param-name))
					    )))))
		   (construct
		    ,@(remove-if #'null
				 (loop for e in members
				       collect
				       (destructuring-bind (&key name type param doc initform param-name member-name) e
					 (cond
					   (param
					    (if (eq name 'callback)
						`(,member-name (std--move ,param-name))
						`(,member-name ,param-name))) 
					   (initform
					    `(,member-name ,initform)))))))
					;(explicit)	    
		   (values :constructor)))

		(defmethod ,(format nil "~~~a" class-name) ()
		  (declare
		   (values :constructor)))


		(defmethod literal (value)
		  (declare (type int value)
			   (values Operation))
		  (let ((op (space Operation (curly ))))
		    (do0 (setf op.kind OperationKind--Literal)
			 (setf op.value value))
		    (return op)))
		
		#+nil
		,@(remove-if
		   #'null
		   (loop for e in members
			 appending
			 (destructuring-bind (&key name type param doc initform param-name member-name) e
			   (let ((get (cl-change-case:camel-case (format nil "get-~a" name)))
				 (set (cl-change-case:camel-case (format nil "set-~a" name)))
				 (const-p (let* ((s  (format nil "~a" type))
						 (last-char (aref s (- (length s)
								       1))))
					    (not (eq #\* last-char)))))
			     `(,(if doc
				    `(doc ,doc)
				    "")
			       (defmethod ,get ()
				 (declare ,@(if const-p
						`((const)
						  (values ,(format nil "const ~a&" type)))
						`((values ,type))))
				 (return ,member-name))
			       (defmethod ,set (,member-name)
				 (declare (type ,type ,member-name))
				 (setf (-> this ,member-name)
				       ,member-name)))))))
		"private:"
		
		,@(remove-if #'null
			     (loop for e in members
				   collect
				   (destructuring-bind (&key name type param doc initform param-name member-name) e
				     (if initform
					 `(space ,type ,member-name (curly ,initform))
					 `(space ,type ,member-name)))))))
      :format t))

  #+nil
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
  
  (let* ((l-prim0 `((:name Add :symbol +) (:name Sub :symbol -) (:name Mul :symbol *) (:name Dup)
		    (:name Drop) (:name Swap) (:name Dot :symbol ".") (:name LessThan :symbol < :short lt) (:name GreaterThan :symbol > :short gt)
		    (:name Equal :symbol = :short eq) (:name Fetch :symbol @) (:name Store :symbol !)))
	 (l-prim (loop for e in l-prim0
		       collect
		       (destructuring-bind (&key
					      name
					      (symbol (string-upcase (format nil "~a" name)))
					      (short (string-downcase (format nil "~a" name)))) e
			 `(:name ,name
			   :symbol ,symbol
			   :short ,short))
	       )))

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
							       ,@(loop for e in l-prim
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
      :format t))

    (write-source
     (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames "helpers.h" *source-dir*))
     `(do0
       (space enum class "Error : int" (curly (comma (= Unknown_Word 1)
						     (= Stack_Error 2)
						     (= Compile_Error 3))))
       (space enum class Primitive
	      (curly
	       ,@(mapcar #'second l-prim)))
       ,@(loop for e in `((:name OperationKind :values (Literal
							Primitive
							CallWord
							If))
			  (:name ParseMode :values (Immediate Definition))
			  (:name SequenceStop :values (End Else Then)))
	       collect
	       (destructuring-bind (&key name values) e
		 `(space enum class ,name
			 (curly ,@values)))))
     :omit-parens t)
    
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

		"constexpr auto kOk = 0;"
		
		
		

		

		"class ForthVM;"
		"using CompiledWord = int (*)(ForthVM*);"

		(defun to_upper (text)
		  (declare (type "std::string_view" text)
			   (values "std::string"))
		  (let ((upper (std--string text))))
		  (std--transform (upper.begin)
				  (upper.end)
				  (upper.begin)
				  (lambda (value)
				    (declare (type "unsigned char" value))
				    (return ("static_cast<char>"
					     (std--toupper value)))))
		  (return upper))

		(defun normalize_dictionary_name (text)
		  (declare (type "std::string_view" text)
			   (values "std::string"))
		  "constexpr std::size_t kMaxNameLength{31};"
		  (let ((normalized (to_upper text)))
		    (when (< kMaxNameLength (normalized.size))
		      (normalized.resize kMaxNameLength))
		    (return normalized)))

		(defun split_on_spaces (line)
		  (declare (type "const std::string&" line)
			   (values "std::vector<std::string>"))
		  (let ((tokens "std::vector<std::string>{}")
			(current "std::string{}"))
		    (for-range (ch line)
			       (declare (type auto ch))
			       (when (== ch (char " "))
				 (unless (current.empty)
				   (tokens.push_back current)
				   (current.clear))
				 continue)
			       (current.push_back ch))
		    (unless (current.empty)
		      (tokens.push_back current))
		    (return tokens)))

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

		(defun to_status (error)
		  (declare (type Error error)
			   (values int))
		  (return ("static_cast<int>" error)))

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
     :tidy nil)))
