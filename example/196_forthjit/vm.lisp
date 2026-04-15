(in-package :cl-cpp-generator2)

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


(let* ((class-name `ForthVM)
       (members `((:name compile_mode_ :type bool :initform false)
		  (:name fuel_ :type int :initform 1000)
		  (:name data_stack_ :type "std::vector<int>")
		  (:name pending_name_ :type "std::string")
		  (:name pending_tokens_ :type "std::vector<std::string>")
		  (:name variables_ :type "std::vector<VariableEntry>")
		  (:name words_ :type "std::vector<WordEntry>")
		  (:name variable_lookup_ :type "std::unordered_map<std::string, int>")
		  (:name word_lookup_ :type "std::unordered_map<std::string, int>"))))
  (write-class
   :dir *full-source-dir*
   :name class-name
   :headers `()
   :header-preamble `(do0 (include<> vector string unordered_map)
			  (include "Operation.h")
			  (include "JITCompiler.h"))
   :implementation-preamble `(do0 (include<> iostream
					     algorithm vector)
				  (include "helpers.h"))

   :code `(do0
	   
	   (defun to_status (error)
	     (declare (type Error error)
		      (values int))
	     (return ("static_cast<int>" error)))

	   
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
	   (defclass ,class-name ()
	     "static constexpr auto MAX_STACK =  256;"
	     "static constexpr auto MAX_DICT =  64;"
	     "static constexpr auto FUEL_LIMIT =  10'000;"
	     
	     "public:"
	     #+nil
	     (space enum Error (curly
				kOk
				Error--Stack_Underflow
				Error--Stack_Overflow
				Error--Dictionary_Full
				Error--Compile_Error
				Error--Invalid_Fuel))
	     
	     (defstruct0 VariableEntry
		 (name "std::string")
	       ("value{0}" int))

	     (defstruct0 WordEntry
		 (name "std::string")
	       (jit_result "gcc_jit_result*")
	       (function CompiledWord))

	     ,@(loop for e in members
		     collect
		     (destructuring-bind (&key name type initform) e
		       (if initform
			   `(space ,type ,name (curly ,initform))
			   `(space ,type ,name))))

	     (defmethod ForthVM ()
	       (declare (values :constructor))
	       )

	     (defmethod ~ForthVM ()
	       (declare (values :constructor))
	       (for-range (word words_)
			  (when (dot word jit_result)
			    (gcc_jit_result_release (dot word jit_result)))))

	     (defmethod process_line (line)
	       (declare (type "const std::string&" line) (values void))
	       (let ((tokens (split_on_spaces line))
		     (idx (cast std--size_t 0)))
		 (when compile_mode_
		   (setf idx (consume_definition_tokens tokens idx))
		   (when (<= (tokens.size) idx)
		     (return)))
		 (while (< idx (tokens.size))
			(let ((upper (to_upper (aref tokens idx))))
			  (cond
			    ((== upper (string ":"))
			     (when (<= (tokens.size) (+ idx 1))
			       (throw Error--Compile_Error))
			     (begin_definition (aref tokens (+ idx 1)))
			     (setf idx (consume_definition_tokens tokens (+ idx 2)))
			     (when compile_mode_
			       (return))
			     continue)
			    ((== upper (string "VARIABLE"))
			     (when (<= (tokens.size) (+ idx 1))
			       (throw Error--Compile_Error))
			     (define_variable (aref tokens (+ idx 1)))
			     (incf idx 2)
			     continue)))
			(let ((start idx))
			  (while (< idx (tokens.size))
				 (let ((current (to_upper (aref tokens idx))))
				   (when (logior (== current (string ":"))
						 (== current (string "VARIABLE")))
				     break))
				 (incf idx))
			  (let ((segment (std--vector (+ (tokens.begin) start) (+ (tokens.begin) idx))))
			    (unless (segment.empty)
			      (execute_segment segment)))))))

	     (defmethod abort_pending_definition ()
	       (declare (values void))
	       (setf compile_mode_ false)
	       (pending_name_.clear)
	       (pending_tokens_.clear))

	     (defmethod push_literal (value)
	       (declare (type int value) (values int))
	       (let ((status (consume_fuel)))
		 (when (!= status kOk) (return status))
		 (return (push_raw value))))

	     ,@(loop for e in `(add sub mul)
		     collect
		     `(defmethod ,e ()
			(declare (values int))
			(let ((status (consume_fuel)))
			  (when (!= status kOk) (return status))
			  (when (< (data_stack_.size) 2) (return (to_status Error--Stack_Error)))
			  (let ((b (data_stack_.back)))
			    (data_stack_.pop_back)
			    (let ((a (data_stack_.back)))
			      (data_stack_.pop_back)
			      (data_stack_.push_back (,(case e (add `+) (sub `-) (mul `*)) a b))))
			  (return kOk))))

	     ,@(remove-if #'null
			  (loop for e in *l-prim*
				collect
				(destructuring-bind (&key name short &allow-other-keys) e
				  (declare (ignore short))
				  (case name
				    ((Add Sub Mul) nil)
				    (t
				     `(defmethod ,(intern (string-upcase (format nil "~a" name))) ()
					(declare (values int))
					(let ((status (consume_fuel)))
					  (when (!= status kOk) (return status))
					  ,(case name
					     (Dup `(progn (when (< (data_stack_.size) 1) (return (to_status Error--Stack_Error)))
							  (data_stack_.push_back (data_stack_.back))))
					     (Drop `(progn (when (< (data_stack_.size) 1) (return (to_status Error--Stack_Error)))
							   (data_stack_.pop_back)))
					     (Swap `(progn (when (< (data_stack_.size) 2) (return (to_status Error--Stack_Error)))
							   (let ((b (data_stack_.back)))
							     (data_stack_.pop_back)
							     (let ((a (data_stack_.back)))
							       (data_stack_.pop_back)
							       (data_stack_.push_back b)
							       (data_stack_.push_back a)))))
					     (Dot `(progn (when (< (data_stack_.size) 1) (return (to_status Error--Stack_Error)))
							  (<< std--cout (data_stack_.back) (string " "))
							  (data_stack_.pop_back)))
					     (LessThan `(progn (when (< (data_stack_.size) 2) (return (to_status Error--Stack_Error)))
							       (let ((b (data_stack_.back)))
								 (data_stack_.pop_back)
								 (let ((a (data_stack_.back)))
								   (data_stack_.pop_back)
								   (data_stack_.push_back (? (< a b) (cast int 1) (cast int 0)))))))
					     (GreaterThan `(progn (when (< (data_stack_.size) 2) (return (to_status Error--Stack_Error)))
								  (let ((b (data_stack_.back)))
								    (data_stack_.pop_back)
								    (let ((a (data_stack_.back)))
								      (data_stack_.pop_back)
								      (data_stack_.push_back (? (> a b) (cast int 1) (cast int 0)))))))
					     (Equal `(progn (when (< (data_stack_.size) 2) (return (to_status Error--Stack_Error)))
							    (let ((b (data_stack_.back)))
							      (data_stack_.pop_back)
							      (let ((a (data_stack_.back)))
								(data_stack_.pop_back)
								(data_stack_.push_back (? (== a b) (cast int 1) (cast int 0)))))))
					     (Fetch `(progn (when (< (data_stack_.size) 1) (return (to_status Error--Stack_Error)))
							    (let ((idx (data_stack_.back)))
							      (data_stack_.pop_back)
							      (when (logand (<= 0 idx) (< idx (cast int (variables_.size))))
								(data_stack_.push_back (dot (aref variables_ idx) value))))))
					     (Store `(progn (when (< (data_stack_.size) 2) (return (to_status Error--Stack_Error)))
							    (let ((idx (data_stack_.back)))
							      (data_stack_.pop_back)
							      (let ((val (data_stack_.back)))
								(data_stack_.pop_back)
								(when (logand (<= 0 idx) (< idx (cast int (variables_.size))))
								  (setf (dot (aref variables_ idx) value) val)))))))
					  (return kOk))))))))

	     (defmethod call_word (index)
	       (declare (type int index) (values int))
	       (let ((status (consume_fuel)))
		 (when (!= status kOk) (return status))
		 (when (logior (< index 0) (<= (cast int (words_.size)) index))
		   (return Error--Dictionary_Full))
		 (return ((dot (aref words_ index) function) (this)))))
	     
	     (defmethod pop_condition (out)
	       (declare (type int* out) (values int))
	       (let ((status (consume_fuel)))
		 (when (!= status kOk) (return status))
		 (when (< (data_stack_.size) 1) (return (to_status Error--Stack_Error)))
		 (setf (deref out) (data_stack_.back))
		 (data_stack_.pop_back)
		 (return kOk)))

	     "protected:"
	     (defmethod consume_fuel ()
	       (declare (values int))
	       (if (<= fuel_ 0)
		   (return Error--Invalid_Fuel))
	       (decf fuel_)
	       (return kOk))

	     (defmethod push_raw (value)
	       (declare (type int value) (values int))
	       (data_stack_.push_back value)
	       (return kOk))
	     
	     (defmethod begin_definition (name)
	       (declare (type "const std::string&" name) (values void))
	       (setf compile_mode_ true
		     pending_name_ name
		     )
	       (pending_tokens_.clear) )

	     (defmethod define_variable (name)
	       (declare (type "const std::string&" name) (values void))
	       (let ((idx (cast int (variables_.size))))
		 (variables_.push_back (curly name 0))
		 (setf (aref variable_lookup_ name) idx)))

	     (defmethod consume_definition_tokens (tokens start_index)
	       (declare (type "const std::vector<std::string>&" tokens)
			(type std--size_t start_index)
			(values "std::size_t"))
	       (let ((i start_index))
		 (while (< i (tokens.size))
			(let ((token (aref tokens i)))
			  (incf i)
			  (when (== (to_upper token) (string ";"))
			    (finish_definition)
			    (return i))
			  (pending_tokens_.push_back token)))
		 (return i)))

	     (defmethod parse_operations (tokens mode)
	       (declare (type "const std::vector<std::string>&" tokens) (type ParseMode mode) (values "std::vector<Operation>") (const))
	       (let ((result (parse_sequence tokens 0 mode)))
		 (when (logior (!= result.stop SequenceStop--End) (!= result.next_index (tokens.size)))
		   (throw Error--Compile_Error))
		 (return result.operations)))

	     (defmethod parse_sequence (tokens index mode)
	       (declare (type "const std::vector<std::string>&" tokens) (type std--size_t index) (type ParseMode mode) (values ParseResult) (const))
	       (let ((result (space ParseResult (curly))))
		 (setf result.next_index index)
		 (while (< result.next_index (tokens.size))
			(let ((token (aref tokens result.next_index))
			      (upper (to_upper token)))
			  (when (== upper (string "ELSE"))
			    (setf result.stop SequenceStop--Else)
			    (incf result.next_index)
			    (return result))
			  (when (== upper (string "THEN"))
			    (setf result.stop SequenceStop--Then)
			    (incf result.next_index)
			    (return result))
			  (when (== upper (string "IF"))
			    (let ((true_branch (parse_sequence tokens (+ result.next_index 1) mode)))
			      (when (== true_branch.stop SequenceStop--End)
				(throw Error--Compile_Error))
			      (let ((false_branch (space "std::vector<Operation>" (curly))))
				(if (== true_branch.stop SequenceStop--Else)
				    (let ((parsed_false (parse_sequence tokens true_branch.next_index mode)))
				      (when (!= parsed_false.stop SequenceStop--Then)
					(throw Error--Compile_Error))
				      (setf false_branch (std--move parsed_false.operations))
				      (setf result.next_index parsed_false.next_index))
				    (setf result.next_index true_branch.next_index))
				(result.operations.push_back (Operation--if_op (std--move true_branch.operations) (std--move false_branch)))
				continue)))
			  (result.operations.push_back (resolve_token token mode))
			  (incf result.next_index)))
		 (return result)))

	     (defmethod resolve_token (token mode)
	       (declare (type "const std::string&" token) (type ParseMode mode) (values Operation) (const))
	       (let ((value (parse_integer token)))
		 (when value
		   (return (Operation--literal (deref value)))))
	       (let ((upper (to_upper token)))
		 (when (logior (== upper (string ":")) (== upper (string ";")) (== upper (string "VARIABLE")))
		   (throw Error--Compile_Error))
		 (let ((primitive (lookup_primitive upper)))
		   (when primitive
		     (return (Operation--primitive_op (deref primitive)))))
		 (let ((name (normalize_dictionary_name token))
		       (variable (variable_lookup_.find name)))
		   (when (!= variable (variable_lookup_.end))
		     (return (Operation--literal (-> variable second)))))
		 (let ((word (word_lookup_.find name)))
		   (when (!= word (word_lookup_.end))
		     (return (Operation--call_word (-> word second)))))
		 (when (== mode ParseMode--Definition)
		   (throw Error--Compile_Error))
		 (throw Error--Unknown_Word)))

	     (defmethod execute_operations (operations)
	       (declare (type "const std::vector<Operation>&" operations) (values int))
	       (for-range (operation operations)
			  (let ((status (execute_operation operation)))
			    (when (!= status kOk)
			      (return status))))
	       (return kOk))

	     (defmethod execute_operation (operation)
	       (declare (type "const Operation&" operation) (values int))
	       (case operation.kind
		 (OperationKind--Literal
		  (return (push_literal operation.value)))
		 (OperationKind--Primitive
		  (return (execute_primitive operation.primitive)))
		 (OperationKind--CallWord
		  (return (call_word operation.value)))
		 (OperationKind--If
		  (let ((condition 0)
			(status (pop_condition (& condition))))
		    (when (!= status kOk)
		      (return status))
		    (if (!= condition 0)
			(return (execute_operations operation.true_branch))
			(return (execute_operations operation.false_branch))))))
	       (return (to_status Error--Compile_Error)))

	     (defmethod execute_primitive (primitive)
	       (declare (type Primitive primitive) (values int))
	       (case primitive
		 ,@(loop for e in *l-prim* collect
			 `(,(format nil "Primitive::~a" (getf e :name)) (return (,(intern (string-upcase (getf e :name))))))))
	       (return (to_status Error--Compile_Error)))

	     (defmethod validate_definition_line (tokens)
	       (declare (type "const std::vector<std::string>&" tokens) (values void) (const))
	       (let ((if_stack (space "std::vector<bool>" (curly))))
		 (for-range (token tokens)
			    (let ((upper (to_upper token)))
			      (when (== upper (string "IF"))
				(if_stack.push_back false)
				continue)
			      (when (== upper (string "ELSE"))
				(when (logior (if_stack.empty) (if_stack.back))
				  (throw Error--Compile_Error))
				(setf (if_stack.back) true)
				continue)
			      (when (== upper (string "THEN"))
				(when (if_stack.empty)
				  (throw Error--Compile_Error))
				(if_stack.pop_back))))
		 (unless (if_stack.empty)
		   (throw Error--Compile_Error))))

	     (defmethod is_reserved_name (name)
	       (declare (type "const std::string&" name) (values bool) (const))
	       (let ((prim (lookup_primitive name)))
		 (return (logior (is_reserved_token name) (prim.has_value)))))

	     (defmethod finish_definition ()
	       (declare (values void))
	       (let ((operations (parse_operations pending_tokens_ 0))
		     (compiler (curly JITCompiler))
		     (symbol_name (+ (string "forth_word_") pending_name_ (string "_") (std--to_string (words_.size))))
		     (result (compiler.compile_word symbol_name operations)))
		 (setf (aref word_lookup_ pending_name_) (cast int (words_.size)))
		 (words_.push_back (curly (= .name pending_name_) (= .jit_result result.jit_result) (= .function result.function)))
		 (setf compile_mode_ false)
		 (pending_name_.clear)
		 (pending_tokens_.clear)))

	     (defmethod execute_segment (tokens)
	       (declare (type "const std::vector<std::string>&" tokens) (values void))
	       (let ((operations (parse_operations tokens ParseMode--Immediate))
		     (status (execute_operations operations)))
		 (when (!= status kOk)
		   (throw ("static_cast<Error>" status)))))

	     (defmethod is_dictionary_full ()
	       (declare (values bool))
	       (let ((MAX_DICT 1000))
		 (return (<= MAX_DICT (+ (variables_.size) (words_.size)))))))
	   
	   "extern \"C\" {"
	   (defun forth_push_literal (vm value) (declare (type "ForthVM*" vm) (type int value) (values int)) (return (-> vm (push_literal value))))
	   (defun forth_add (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (add))))
	   (defun forth_sub (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (sub))))
	   (defun forth_mul (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (mul))))
	   (defun forth_dup (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (dup))))
	   (defun forth_drop (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (drop))))
	   (defun forth_swap (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (swap))))
	   (defun forth_dot (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (execute_primitive Primitive--Dot))))
	   (defun forth_lt (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (lessthan))))
	   (defun forth_gt (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (greaterthan))))
	   (defun forth_eq (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (equal))))
	   (defun forth_fetch (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (fetch))))
	   (defun forth_store (vm) (declare (type "ForthVM*" vm) (values int)) (return (-> vm (store))))
	   (defun forth_pop_condition (vm out_condition) (declare (type "ForthVM*" vm) (type "int*" out_condition) (values int)) (return (-> vm (pop_condition out_condition))))
	   (defun forth_call_word (vm word_index) (declare (type "ForthVM*" vm) (type int word_index) (values int)) (return (-> vm (call_word word_index))))
	   "}")
   :format t))
