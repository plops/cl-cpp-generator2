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
				      (return tokens))))
   :code `(do0
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

	     (defmethod execute_line (line)
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
			  (when (< (data_stack_.size) 2) (return Error--Stack_Underflow))
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
				     `(defmethod ,(string-downcase (format nil "~a" name)) ()
					(declare (values int))
					(let ((status (consume_fuel)))
					  (when (!= status kOk) (return status))
					  ,(case name
					     (Dup `(progn (when (< (data_stack_.size) 1) (return Error--Stack_Underflow))
							  (data_stack_.push_back (data_stack_.back))))
					     (Drop `(progn (when (< (data_stack_.size) 1) (return Error--Stack_Underflow))
							   (data_stack_.pop_back)))
					     (Swap `(progn (when (< (data_stack_.size) 2) (return Error--Stack_Underflow))
							   (let ((b (data_stack_.back)))
							     (data_stack_.pop_back)
							     (let ((a (data_stack_.back)))
							       (data_stack_.pop_back)
							       (data_stack_.push_back b)
							       (data_stack_.push_back a)))))
					     (Dot `(progn (when (< (data_stack_.size) 1) (return Error--Stack_Underflow))
							  (<< std--cout (data_stack_.back) (string " "))
							  (data_stack_.pop_back)))
					     (LessThan `(progn (when (< (data_stack_.size) 2) (return Error--Stack_Underflow))
							       (let ((b (data_stack_.back)) (a (progn (data_stack_.pop_back) (data_stack_.back))))
								 (data_stack_.pop_back)
								 (data_stack_.push_back (if (< a b) (cast int 1) (cast int 0))))))
					     (GreaterThan `(progn (when (< (data_stack_.size) 2) (return Error--Stack_Underflow))
								  (let ((b (data_stack_.back)) (a (progn (data_stack_.pop_back) (data_stack_.back))))
								    (data_stack_.pop_back)
								    (data_stack_.push_back (if (> a b) (cast int 1) (cast int 0))))))
					     (Equal `(progn (when (< (data_stack_.size) 2) (return Error--Stack_Underflow))
							    (let ((b (data_stack_.back)) (a (progn (data_stack_.pop_back) (data_stack_.back))))
							      (data_stack_.pop_back)
							      (data_stack_.push_back (if (== a b) (cast int 1) (cast int 0))))))
					     (Fetch `(progn (when (< (data_stack_.size) 1) (return Error--Stack_Underflow))
							    (let ((idx (data_stack_.back)))
							      (data_stack_.pop_back)
							      (when (logand (<= 0 idx) (< idx (cast int (variables_.size))))
								(data_stack_.push_back (dot (aref variables_ idx) value))))))
					     (Store `(progn (when (< (data_stack_.size) 2) (return Error--Stack_Underflow))
							    (let ((idx (data_stack_.back)) (val (progn (data_stack_.pop_back) (data_stack_.back))))
							      (data_stack_.pop_back)
							      (when (logand (<= 0 idx) (< idx (cast int (variables_.size))))
								(setf (dot (aref variables_ idx) value) val))))))
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
		 (when (< (data_stack_.size) 1) (return Error--Stack_Underflow))
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

	     (defun parse_operations (tokens mode)
	       (declare (static) (type "const std::vector<std::string>&" tokens) (type "int" mode) (values "std::vector<Operation>"))
	       (let ((res (curly)))
		 (return res)))

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
	       (for-range (token tokens)
			  (let ((upper (to_upper token)))
			    (cond
			      ((variable_lookup_.count upper)
			       (push_literal (aref variable_lookup_ upper)))
			      ((word_lookup_.count upper)
			       (call_word (aref word_lookup_ upper)))
			      (t 
			       (try
				(let ((val (std--stoi token)))
				  (push_literal val))
				(catch (t ()
					  (std--cerr (string "Unknown word: ") token (std--endl))))))))))

	     (defmethod is_dictionary_full ()
	       (declare (values bool))
	       (let ((MAX_DICT 1000))
		 (return (<= MAX_DICT (+ (variables_.size) (words_.size))))))))
   :format t))
