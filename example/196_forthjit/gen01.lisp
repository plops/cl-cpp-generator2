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
  
  (load "util.lisp")
  
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
		#+nil
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
  (let ((l-prim `(Add Sub Mul Dup Drop Swap Dot LessThan GreaterThan Equal Fetch Store))) 
   (write-source
    (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames "main.cpp" *source-dir*))
    `(do0
      (include<> algorithm
		 array
		 charconv
		 cctype
		 cstdblib
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
      "using namespace gccjit;"
      (space namespace
	     (progn

	       "constexpr auto kOk = 0;"

	       (space enum class Error-int (curly (comma (= Unknown_Word 1)
							 (= Stack_Error 2)
							 (= Compile_Error 3))))
	       (space enum class Primitive
		      (curly
		       ,@l-prim))


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

	     
	       ))
     
     
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
