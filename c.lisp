(declaim (optimize
	  (safety 3)
	  (speed 0)
	  (debug 3)))
#-nil
(progn (ql:quickload "alexandria")
       (defpackage :cl-cpp-generator2
	 (:use :cl
	  :alexandria
	       :cl-ppcre)
	 (:export
	  #:write-source)))
;;(setf *features* (union *features* '(:generic-c)))
;;(setf *features* (set-difference *features* '(:generic-c)))
(in-package :cl-cpp-generator2)

(setf (readtable-case *readtable*) :invert)

(defparameter *file-hashes* (make-hash-table))
(defparameter *auto-keyword* "auto")

(defun write-notebook (&key nb-file nb-code (std 17))
  "write xeus-cling C++ jupyter notebook"
  (let ((tmp (format nil "~a.tmp" nb-file)))
    (with-output-to-file (s tmp :if-exists :supersede
				:if-does-not-exist :create)
      (format
       s "~a~%"
       (jonathan:to-json
	`(:|cells|
	   ,(loop for e in nb-code
		  collect
		  (destructuring-bind (name &rest rest) e
		    (case name
		      (`markdown `(:cell_type "markdown"
				   :metadata :empty
				   :source
				   ,(loop for p in rest
					  collect
					  (format nil "~a~c" p #\Newline))))
		      (`cpp `(:cell_type "code"
			      :metadata :empty
			      :execution_count :null
			      :outputs ()
			      :source
			      ,(loop for p in rest
				     appending
				     (let ((tempfn #+sbcl "/dev/shm/cell.cpp"
						   #+ecl (format nil "~a_tmp_cell" nb-file)))
				       (write-source tempfn p)
				       (with-open-file (stream (format nil "~a" tempfn))
					 (loop for line = (read-line stream nil)
					       while line
					       collect
					       (format nil "~a~c" line #\Newline)))))))
		      )))
	   :|metadata| (:|kernelspec| (:|display_name| ,(format nil "C++~a" std)
					:|language| ,(format nil "C++~a" std)
				       :|name| ,(format nil "xcpp~a" std)))
	   :|nbformat| 4
	   :|nbformat_minor| 2

	   #+nil
	   (:metadata (:kernelspec (:display_name "Python 3"
				    :language "python"
				    :name "python3"))
	    :nbformat 4
	    :nbformat_minor 2)))))
    #+nil
    (sb-ext:run-program "/usr/bin/python3" `("-mjson.tool" ,nb-file))
    #-sbcl
    (external-program:run
     "/usr/bin/jq"
     `("-M" "." ,tmp)
     :output nb-file
     :if-output-exists :supersede
     )
    #+sbcl
    (sb-ext:run-program "/usr/bin/jq" `("-M" "." ,tmp)
			:output nb-file
			:if-output-exists :supersede)
    (delete-file tmp)))



(defun write-source (name code &key
				 (dir (user-homedir-pathname))
				 ignore-hash
				 (format t)
				 (tidy t)
				 (omit-parens nil)
				 )
	"This function writes the given code into a file specified by the name and directory.
	 It also provides options to control the behavior of the writing process.

	 Parameters:
		 name - the name of the file to be written
		 code - the code to be written into the file (s-expression)
		 dir - the directory where the file will be written (default is user's home directory)
		 ignore-hash - a flag indicating whether to ignore the hash value of the file, which means the code will be written into the file if the hash value is different from the previous one (default is nil)
		 format - a flag indicating whether to format the code using clang-format (default is t)
		 tidy - a flag indicating whether to tidy the code using clang-tidy (default is t)
		 omit-parens - a flag indicating whether to omit redundant parentheses in the code (default is nil)

	 Example usage:
		 (write-source \"myfile.cpp\" `(defun foo () (declare (values int)) (return 42)) :dir \"/path/to/directory\" :format nil)"


  ;(format t "<write-source code='~a'>~%" code)
  (let* ((fn (merge-pathnames (format nil "~a" name)
			      dir))
	 (code-str (m-of (emit-c :code code :header-only nil
				 :omit-redundant-parentheses omit-parens)))
	 (fn-hash (sxhash fn))
	 (code-hash (sxhash code-str)))

    (multiple-value-bind (old-code-hash exists) (gethash fn-hash *file-hashes*)
      (when (or (not exists) ignore-hash (/= code-hash old-code-hash)
		(not (probe-file fn)))
	(format t "write code into file: '~a'~%" fn)
	;; store the sxhash of the c source in the hash table
	;; *file-hashes* with the key formed by the sxhash of the full
	;; pathname
	(setf (gethash fn-hash *file-hashes*) code-hash)
	(with-open-file (s fn
			   :direction :output
			   :if-exists :supersede
			   :if-does-not-exist :create)
	  (write-sequence code-str s))
	;; https://travisdowns.github.io/blog/2019/11/19/toupper.html
	;; header reordering can affect compilation performance
	;; https://stackoverflow.com/questions/60334299/clang-format-disable-ordering-includes
	;; SortIncludes: false    in .clang-format
	(when format
	  (sb-ext:run-program "/usr/bin/clang-format"
			      (list "-i"  (namestring fn)
				    "-style=llvm" ;; removes unneccessary parentheses (i hope)
				    ;; "-style='{PenaltyReturnTypeOnItsOwnLine: 100000000}'"
				    )))
	(when tidy
	  (sb-ext:run-program "/usr/bin/clang-tidy"
			      (list (namestring fn)
       				    "--checks=readability-*"
				    "--fix"
				    ))
	  )))))

;; http://clhs.lisp.se/Body/s_declar.htm
;; http://clhs.lisp.se/Body/d_type.htm

;; go through the body until no declare anymore



;; (declare (type int a b) (type float c)
;; (declare (values int &optional))
;; (declare (values int float &optional))

;; FIXME doesnt handle documentation strings



;; FIXME: misses const override
(defun consume-declare (body)
  "Take a list of instructions from `body`, parse type declarations,
return the `body` without them and a hash table with an environment. The
entry `return-values` contains a list of return values. Currently supports
type, values, construct, and capture declarations. Construct is member 
assignment in constructors. Capture is for lambda functions.

Parameters:
- `body` (list): The list of instructions to process.

Returns:
- `new-body` (list): The modified `body` without the type declarations.
- `env` (hash-table): The hash table containing the environment.
- `captures` (list): The list of captured variables.
- `constructs` (list): The list of constructed variables.
- `const-p` (boolean): Indicates if the `const` declaration is present.
- `explicit-p` (boolean): Indicates if the `explicit` declaration is present.
- `inline-p` (boolean): Indicates if the `inline` declaration is present.
- `static-p` (boolean): Indicates if the `static` declaration is present.
- `virtual-p` (boolean): Indicates if the `virtual` declaration is present.
- `noexcept-p` (boolean): Indicates if the `noexcept` declaration is present.
- `final-p` (boolean): Indicates if the `final` declaration is present.
- `override-p` (boolean): Indicates if the `override` declaration is present.
- `pure-p` (boolean): Indicates if the `pure` declaration is present.
- `template` (symbol): The template declaration.
- `template-instance` (symbol): The template instance declaration."
  (let ((env (make-hash-table))
	(captures nil)
	(constructs nil)
	(const-p nil)
	(explicit-p nil)
	(inline-p nil)
	(static-p nil)
	(virtual-p nil)
	(noexcept-p nil)
	(final-p nil)
	(override-p nil)
	(pure-p nil)
	(template nil)
	(template-instance nil)
	(looking-p t)
	(new-body nil))
    (loop for e in body do
      (if looking-p
	  (if (listp e)
	      (if (eq (car e) 'declare)
		  (loop for declaration in (cdr e) do
		    (when (eq (first declaration) 'type)
		      (destructuring-bind (symb type &rest vars) declaration
			(declare (ignorable symb))
			(loop for var in vars do
			  (setf (gethash var env) type))))
		    (when (eq (first declaration) 'capture)
		      (destructuring-bind (symb &rest vars) declaration
			(declare (ignorable symb))
			(loop for var in vars do
			  (push var captures))))

		    (when (eq (first declaration) 'construct)
		      (destructuring-bind (symb &rest vars) declaration
			(declare (ignorable symb))
			(loop for var in vars do
			  (push var constructs))))
		    (when (eq (first declaration) 'const)
		      (setf const-p t))
		    (when (eq (first declaration) 'explicit)
		      (setf explicit-p t))
		    (when (eq (first declaration) 'inline)
		      (setf inline-p t))
		    (when (eq (first declaration) 'virtual)
		      (setf virtual-p t))
		    (when (eq (first declaration) 'noexcept)
			  (setf noexcept-p t))
		    (when (eq (first declaration) 'final)
			  (setf final-p t))
		    (when (eq (first declaration) 'override)
		      (setf override-p t))
		    (when (eq (first declaration) 'pure)
		      (setf pure-p t))
		    (when (eq (first declaration) 'static)
		      (setf static-p t))
		    (when (eq (first declaration) 'template)
		      (setf template (second declaration)))
		    (when (eq (first declaration) 'template-instance)
		      (setf template-instance (second declaration)))
		    (when (eq (first declaration) 'values)
		      (destructuring-bind (symb &rest types-opt) declaration
			(declare (ignorable symb))
			;; if no values specified parse-defun will emit void
			;; if (values :constructor) then nothing will be emitted
			(let ((types nil))
			  ;; only collect types until occurrence of &optional
			  (loop for type in types-opt do
			    (unless (eq #\& (aref (format nil "~a" type) 0))
			      (push type types)))
			  (setf (gethash 'return-values env) (reverse types))))))
		  (progn
		    (push e new-body)
		    (setf looking-p nil)))
	      (progn
		(setf looking-p nil)
		(push e new-body)))
	  (push e new-body)))
    (values (reverse new-body) env (reverse captures) (reverse constructs)
	    const-p explicit-p inline-p static-p virtual-p noexcept-p final-p override-p pure-p template template-instance)))

(defun lookup-type (name &key env)
	"Get the type of a variable from an environment.

	This function takes a variable name and an environment and returns the type of the variable.
	
	Parameters:
		- name: The name of the variable.
		- env: The environment containing the variable.

	Returns:
		The type of the variable, or nil if the variable is not found in the environment."
	(gethash name env))

(defun variable-declaration (&key name env emit)
	"Find the type of variable NAME in environment ENV and emit the type and name as a 
concatenated string using EMIT. If the variable is not present in the environment,
emit 'auto'. If the type is an array, emit a string 'type name[dimension]'.

Parameters:
- NAME: The name of the variable.
- ENV: The environment in which to look up the variable.
- EMIT: A function used to emit the type and name as a string.

Returns:
A string representing the variable declaration."

	(let* ((type (lookup-type name :env env)))
	  (cond ((null type)
		 (format nil "~a ~a"
			 *auto-keyword*
			 (funcall emit name)))
		((and (listp type)
		      (eq 'array (first type)))
		 (progn
		   ;; array
		   (destructuring-bind (array_ element-type &rest dims) type
		     (assert (eq array_ 'array))
		     (format nil "~a ~a~{[~a]~}"
			     (funcall emit element-type)
			     (funcall emit name)
			     (mapcar emit dims)))))
		(t (format nil "~a ~a"
			   (if type
			       (funcall emit type)
			       *auto-keyword*)
			   (funcall emit name))))))
    #+nil (if (listp type)
	      (if (null type)
		  (format nil "~a ~a"
			  #+generic-c "__auto_type"
			  #-generic-c "auto"
			  (funcall emit name))
		  (progn
		    ;; array
		    (destructuring-bind (array_ element-type &rest dims) type
		      (assert (eq array_ 'array))
		      (format nil "~a ~a~{[~a]~}"
			      element-type
			      (funcall emit name)
			      (mapcar emit dims)))))
	      (format nil "~a ~a"
		      (if type
			  (funcall emit type)
			  #+generic-c "__auto_type"
			  #-generic-c "auto"
			  )
		      (funcall emit name)))

(defun parse-let (code emit &key const decltype)
  "Parse a Common Lisp LET form and emit similar C++ code.

  This function takes a Common Lisp LET form and generates equivalent
  C++ code. The LET form consists of variable declarations and an
  optional DECLARE form. The DECLARE form can be used to declare types
  for the variables. If types are not declared, the 'auto' keyword
  will be used in C++.

  Initial values for the variables are assigned using the C++ initializer list. For example, the input code '(let ((a (std--vector<int> (curly 1 2 3)))))' will generate the output 'auto a{std::vector<int>{1, 2, 3}};'.

  The supported grammar for the LET form is as follows:
  let ({var | (var [init-form])}*) declaration* form*

  Parameters:
    - code: The Common Lisp LET form to parse.
    - emit: The function used to emit child forms below the LET form as C++ code.
    - const: Write const in front of every definition (this is used in letc)
    - decltype: Write decltype(<init-form>) for every definition (this is used in letd)
  Returns:
    The generated C++ code as a string."
  (destructuring-bind (decls &rest body) (cdr code)
    (multiple-value-bind (body env captures constructs const-p explicit-p inline-p static-p virtual-p noexcept-p final-p override-p pure-p template template-instance) (consume-declare body)
      (with-output-to-string (s)
	(format
	 s "~a"
	 (funcall
	  emit
	  `(do0
	    ,@(loop for decl in decls
		    collect
		    (if (listp decl) ;; split into name and initform
			(destructuring-bind (name &optional value) decl
			  ;; FIXME: introducing initializer lists is better for C++ but not working with GLSL (and possibly C)
			  (format nil ;"~a ~@[ = ~a~];"
				  "~a~a ~@[{~a}~];"
				  (if const "const " "")
				  (if decltype
				      (if value
				       (format nil "decltype(~a) ~a"
					       (funcall emit value)
					       name)
				       (break "decltype needs value"))
				      (variable-declaration :name name :env env :emit emit))
				  (unless decltype
				    (when value
				      (funcall emit value)))))
			(format nil "~a;"
				(variable-declaration :name decl :env env :emit emit))))
	    ,@body)))))))

(defun parse-defun (code emit &key header-only (class nil))
	"Emit a C++ function definition or declaration from a Common Lisp DEFUN form.
	
	Arguments:
	- CODE: The DEFUN form to parse.
	- EMIT: The function to emit the C++ code.
	- HEADER-ONLY: If true, the function will be emitted as a declaration only.
	- CLASS: The class name if the function is a member of a class.

	Returns:
	- The C++ function definition or declaration as a string.

	Example:
	(defun foo (a b)
		(declare (values int) inline (type int a b))
		(return (+ a b)))
	will be emitted as 'inline int foo(int a, int b) { return a + b; }'

	Supported grammar: defun function-name lambda-list [declaration*] form*"
  (destructuring-bind (name lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env captures constructs const-p explicit-p inline-p static-p virtual-p noexcept-p final-p override-p pure-p template template-instance) (consume-declare body) ;; py
      (multiple-value-bind (req-param opt-param res-param
			    key-param other-key-p
			    aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(with-output-to-string (s)


	  ;;         template          static          inline  virtual ret   params     pure      override         header-only
	  ;;                                   explicit                   name  const                                        constructs
	  ;;         1                 2       3       4       5       6  7  8  9a      9b        9c               10        11
	  (format s "~%~@[template<~a> ~]~@[~a ~]~@[~a ~]~@[~a ~]~@[~a ~]~a ~a ~a ~@[~a~] ~:[~;=0~] ~:[~;noexcept~] ~:[~;final~] ~:[~;override~] ~:[~;;~]  ~@[: ~a~]"
		  ;; 1 template
		  (when template
		    template)
		  ;; 2 static
		  (when (and static-p
			     header-only)
		    "static")
		  ;; 3 explicit
		  (when (and explicit-p
			     header-only)
		    "explicit")
		  ;; 4 inline
		  (when (and inline-p
			     header-only)
		    "inline")
		  ;; 5 virtual
		  (when (and virtual-p
			     header-only
			     )
					;(format t "virtual defun~%")
		    "virtual")

		  ;; 6 return value
		  (let ((r (gethash 'return-values env)))
		    (if (< 1 (length r))
			(break "multiple return values unsupported: ~a"
			       r)
			(if (car r)
			    (case (car r)
			      (:constructor "") ;; (values :constructor) will not print anything
			      (t (car r)))
			    "void")))
		  ;; 7 function-name, add class if not header
		  (if class
		      (if header-only
			  name
			  (format nil "~a::~a" class name))
		      name)

		  ;; 8 positional parameters, followed by key parameters
		  (funcall emit `(paren
				  ;; positional
				  ,@(loop for p in req-param collect
							     (format nil "~a ~a"
								     (let ((type (gethash p env)))
								       (if type
									   (funcall emit type)
									   (break "can't find type for positional parameter ~a in defun"
										  p)))
								     p))
				  ;; key parameters
				  ;; http://www.crategus.com/books/alexandria/pages/alexandria.0.dev_fun_parse-ordinary-lambda-list.html
				  ,@(loop for ((keyword-name name) init supplied-p) in key-param collect
												 (progn
												   #+nil (format t "~s~%" (list (loop for k being the hash-keys in env using (hash-value v) collect
																							    (format nil "'~a'='~a'~%" k v)) :name name :keyword-name keyword-name :init init))
												   (format nil "~a ~a ~@[~a~]"
													   (let ((type (gethash name env)))
													     (if type
														 (funcall emit type)
														 (break "can't find type for keyword parameter ~a in defun"
															name)))
													   name
													   
													   (when header-only ;; only in class definition
													     (format nil "= ~a" (funcall emit init))))))
				  ))
		  ;; 9 const keyword / or '=0' for pure function
		  (when const-p #+nil
			(and const-p
			     (not header-only))
			"const")

		  nil ;; pure not applicable
		  noexcept-p
		  final-p
		  nil ;; override-p not applicable
		  ;; 10 semicolon if header only
		  header-only
		  ;; 11 constructor initializers
		  (when (and constructs
			     (not header-only))
		    (funcall emit `(comma ,@(mapcar emit constructs)))))
	  (unless header-only
	    (format s "~a~%" (funcall emit `(progn ,@body)))))))))

;; https://stackoverflow.com/questions/21577466/the-order-of-override-and-noexcept-in-the-standard
;; This states that override and final have to come after noexcept.
;; void f() final is the same as virtual f() final override

(defun parse-defmethod (code emit &key header-only (class nil) (in-class-p nil))
	"Emit a C++ class member function definition or declaration from a
Common Lisp DEFMETHOD form.
	
	This function takes a DEFMETHOD form in Common Lisp and
	generates the corresponding C++ class member function
	definition or declaration. The generated code can be emitted
	using the provided EMIT function.

	Arguments:
	- CODE: The DEFMETHOD form to parse.
	- EMIT: The function to emit the C++ code.
	- HEADER-ONLY: If true, the function will be emitted as a declaration only.
	- CLASS: The class name if the function is a member of a class.
	- IN-CLASS-P: Selects if a declaration is emitted inside of a
          class (e.g. in the header) or if a function definition is
          written in an implementation .cpp file. In the latter case,
          the class name is prepended to the function name.

	Returns:
	- The C++ function definition or declaration as a string.

	Example:
        (defclass+ A ()
         \"public:\"
	(defmethod foo (a b)
		(declare (values int) (type int a b))
		(return (+ a b))))
	will be emitted as 'int A::foo(int a, int b) { return a + b; }'

	Supported grammar: defmethod function-name lambda-list [declaration*] form*"

  (destructuring-bind (name lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env captures constructs const-p explicit-p inline-p static-p virtual-p noexcept-p final-p override-p pure-p template template-instance) (consume-declare body) ;; py
      (multiple-value-bind (req-param opt-param res-param
			    key-param other-key-p
			    aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(when (and (or (and inline-p
			    (not  (eq in-class-p 'defclass+)))
		       pure-p
		       )
		   (not header-only))
	  (return-from parse-defmethod ""))

	#+nil
	(when (and virtual-p
		   (not header-only))
	  (return-from parse-defmethod (format nil "// virtual method ~a" name)))

	(with-output-to-string (s)
	  ;;         template          static          inline  virtual ret   params               noexc,fina         header-only
	  ;;                                   explicit                   name  const   pure      override                       constructs
	  ;;         1                 2       3       4       5       6  7  8  9       9b        9c               10               11
					; format s "~@[template<~a> ~]~@[~a ~]~@[~a ~]~@[~a ~]~@[~a ~]~a ~a ~a ~@[~a~] ~:[~;=0~] ~:[~;;~]  ~@[: ~a~]"

	  (format s "~@[template<~a> ~]~@[~a ~]~@[~a ~]~@[~a ~]~@[~a ~]~a ~a ~a ~@[~a~] ~:[~;=0~] ~:[~;noexcept~] ~:[~;final~] ~:[~;override~]  ~:[~;;~]  ~@[: ~a~]"
		  ;; 1 template
		  (when template
		    template)
		  ;; 2 static
		  (when (and static-p
			     (or  (eq in-class-p 'defclass+)
			      header-only))
		    "static")
		  ;; 3 explicit
		  (when (and explicit-p
			     (or (eq in-class-p 'defclass+)
				 header-only))
		    "explicit")
		  ;; 4 inline
		  (when (and inline-p
			     (or (eq in-class-p 'defclass+)
				 header-only))
		    "inline")
		  ;; 5 virtual
		  (when (and virtual-p
			     (not (eq in-class-p 'defclass-cpp))
			     #+nil (or
					;(eq in-class-p 'defclass+)
					;(eq in-class-p 'defclass-hpp)
				    )

					;(or in-class-p header-only)
			     )
					;(format t "virtual defmethod~%")
		    "virtual")

		  ;; 6 return value
		  (let ((r (gethash 'return-values env)))
		    (if (< 1 (length r))
			(break "multiple return values unsupported: ~a"
			       r)
			(if (car r)
			    (case (car r)
			      (:constructor "") ;; (values :constructor) will not print anything
			      (t (car r)))
			    "void")))
		  ;; 7 function-name, add class if not header
		  (if class
		      (if header-only
			  name
			  (format nil "~a~@[< ~a >~]::~a" class template-instance name))
		      name)

		  ;; positional parameters, followed by key parameters
		  (funcall emit `(paren
				  ;; positional
				  ,@(loop for p in req-param collect
							     (format nil "~a ~a"
								     (let ((type (gethash p env)))
								       (if type
									   (funcall emit type)
									   (break "can't find type for positional parameter ~a in defun"
										  p)))
								     p))
				  ;; key parameters
				  ;; http://www.crategus.com/books/alexandria/pages/alexandria.0.dev_fun_parse-ordinary-lambda-list.html
				  ,@(loop for ((keyword-name name) init supplied-p) in key-param collect
												 (progn
												   #+nil (format t "~s~%" (list (loop for k being the hash-keys in env using (hash-value v) collect
																							    (format nil "'~a'='~a'~%" k v)) :name name :keyword-name keyword-name :init init))
												   (format nil "~a ~a ~@[~a~]"
													   (let ((type (gethash name env)))
													     (if type
														 (funcall emit type)
														 (break "can't find type for keyword parameter ~a in defun"
															name)))
													   name
													   (when header-only ;; only in class definition
													     (format nil "= ~a" (funcall emit init))))))
				  ))
		  ;; const keyword
		  (when const-p #+nil
			(and const-p
			     (not header-only))
			"const")
		  (when header-only pure-p)
		  noexcept-p
		  ;(when header-only final-p)
		  (cond (header-only final-p)
			((eq in-class-p 'defclass+)
			 final-p))
		  (cond (header-only override-p)
			((eq in-class-p 'defclass+)
			 override-p))
		  ;(when header-only override-p) ;; FIXME: not working in defclass+
		  ;; semicolon if header only
		  (and (not inline-p) header-only)
		  ;; constructor initializers
		  (when (and constructs
			     (not header-only))
		    (funcall emit `(comma ,@(mapcar emit (loop for (var init) in  constructs
							       collect
							       `(space ,var (curly ,init))))))))
	  (when (or inline-p (not header-only))
	    (format s "~a" (funcall emit `(progn ,@body)))))))))

(defun parse-lambda (code emit)
  "Parse a Common Lisp LAMBDA form and emit similar C++ code.

  This function takes a Common Lisp LAMBDA form and generates
  equivalent C++ code.
  
  Arguments:
    - code: The Common Lisp LAMBDA form to parse.
    - emit: A function used to emit C++ code.

  Returns:
    A string containing the generated C++ code.

  Supported Grammar:
    lambda lambda-list [declaration*] form*

  Example without return value:
    [] (int a, float b) { body }

  Example with return type declaration:
    [] (int a, float b) -> float { body }

  Support for captures:
    Captures can be specified using the (declare (capture ...))
    syntax. Captures are placed into the first set of brackets in the
    generated C++ code.

  Parameters:
    - lambda-list: The lambda list of the Common Lisp LAMBDA form.
    - body: The body of the Common Lisp LAMBDA form.

  Returns:
    A string containing the generated C++ code."
 
  (destructuring-bind (lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env captures constructs const-p) (consume-declare body)
      ;; empty captures shall default to "&"
      (when (null captures)
	(setf captures `("&")))
      (multiple-value-bind (req-param opt-param res-param
			    key-param other-key-p
			    aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(with-output-to-string (s)
	  (format s "[~{~a~^,~}] ~a~@[-> ~a ~]"
		  (mapcar emit captures)
		  (if (null req-param)
		      ""
		      (funcall emit `(paren
				      ,@(loop for p in req-param collect
								 (format nil "~a ~a"
									 (let ((type (gethash p env)))
									   (if type
									       (funcall emit type)
									       (progn
										 ;; (break "can't find type for ~a in defun" p)
										 "auto"
										 )))
									 p
									 )))))
		  (let ((r (gethash 'return-values env)))
		    (if (< 1 (length r))
			(funcall emit `(paren ,@r))
			(car r))))
	  (format s "~a" (funcall emit `(progn ,@body))))))))

(defun print-sufficient-digits-f32 (f)
	"Prints a single floating point number as a string with a given number
of digits. Parses it again and increases the number of digits until
the same bit pattern is obtained.
	 
	 Args:
		 f: The floating point number to be printed.
	 
	 Returns:
		 The string representation of the floating point number with sufficient digits."
	(let* ((a f)
				 (digits 1)
				 (b (- a 1)))
		(unless (= a 0)
			(loop while (and (< 1e-6 (/ (abs (- a b))
				  (abs a)))
		       (< digits 30))
						do
							 (setf b (read-from-string (format nil "~,v,,,,,'eG"
					;"~,vG"
						 digits a
						 )))
							 (incf digits)))
					;(format nil "~,vG" digits a)
					;(format nil "~,v,,,,,'eGf" digits a)
		(let ((str
	    (format nil "~,v,,,,,'eG" digits a)))
			(format nil "~aF" (string-trim '(#\Space) str))
			#+nil
			(if (find #\e str)
	  str
	  (format nil "~af" (string-trim '(#\Space) str))))))


(defun print-sufficient-digits-f64 (f)
	"Prints a double floating point number as a string with a given number
of digits.

	 Parses it again and increases the number of digits until the
	 same bit pattern of the 64-bit float is obtained.

	 Args:
	 - f: The double floating point number to be printed.

	 Returns:
	 - The string representation of the double floating point number with sufficient digits."
	(let* ((a f)
				 (digits 1)
				 (b (- a 1)))
		(unless (= a 0)
			(loop while (and (< 1d-12
			  (/ (abs (- a b))
			     (abs a))
			  )
		       (< digits 30)) do
			 (setf b (read-from-string (format nil "~,vG" digits a)))
			 (incf digits)))
					;(format t "~,v,,,,,'eG~%" digits a)
		(format nil "~,v,,,,,'eG" digits a)
		;(substitute #\e #\d (format nil "~,vG" digits a))
		))
(defparameter *operators*
	`(comma semicolon space space-n comments paren* paren angle bracket curly designated-initializer new indent split-header-and-code do0 pragma include include<> progn namespace do defclass+ defclass protected public defmethod defun defun* defun+ return co_return co_await co_yield throw cast let setf not bitwise-not deref ref + - * ^ xor & / or and logior logand = /= *= ^= <= < != == % << >> incf decf string string-r string-u8 char hex ? if when unless if-constexpr dot aref -> lambda case for for-range dotimes foreach while deftype struct defstruct0 handler-case)
  "This variable stores a list of operators that are supported by the EMIT-C function.
 It is used in the PAREN* form to determine whether parentheses are needed.")


;; https://en.cppreference.com/w/cpp/language/operator_precedence
;; FIXME: how to handle Associativity (right-to-left or
;; left-to-right). do i need a table or is this implicitly handled in
;; emit-c for each symbol itself?
#+nil
(defparameter *precedence* `(#+nil
			     ("::")
			     (hex)
			     (char)
			     (string)
			     ((paren) l) ;; ?? does that go here
			     
			     ((	       ; incf decf (only with a++ a--)
			       ;; unary+ unary-
			       aref	; call cast

			       -> dot 
			       ) l)
					; ((prefix++ prefix--) r)
			     
			     (-unary
			      not bitwise-not
			      cast ref
			      deref
			      sizeof
			      co_await
			      new
			      new[]
			      delete
			      delete[]
			      )
			     #+nil (.* ->*)
			     
			     (*  / %)
			     
			     (+ -)
			     (<< >>)
			     (<=>)
			     (< <= > >=)
			     ;; 10
			     
			     (== !=)
			     ;; 11
			     (and &)
			     (xor ^)
			     (or )
			     
			     (logand &&)
			     ;; 15
			     (logior )

			     ;; 16
			     (? )
			     (throw )
			     (co_yield)
			     (setf =)
			     (incf decf)
			     (*= /= %=)
			     (<<= >>=)
			     (&= ^-	; |=
				 )
			     
			     
			     (comma )
			     ))

(defparameter *precedence* `((:op (scope))
			     (:op (hex))
			     (:op (char))
			     (:op (string))
			     (:op (paren
				   paren*
				   curly
				   aref dot ->) :assoc l) 
			     (:op (-unary
				   not bitwise-not
				   cast
				   deref
				   ref
				   sizeof
				   co_await
				   new
				   new[]
				   delete
				   delete[]
				   )
				  :assoc r)
			     #+nil (.* ->*) ;; pointer to member
			     (:op (*  / %) :assoc l)
			     (:op (+ -) :assoc l)
			     (:op (<< >>) :assoc l)
			     (:op (<=>) :assoc l)
			     (:op (< <= > >=) :assoc l)
			     ;; 10
			     (:op (== !=) :assoc l)
			     ;; 11
			     (:op (and &) :assoc l)
			     (:op (xor ^) :assoc l) 
			     (:op (or) :assoc l)
			     (:op (logand &&) :assoc l) ;; FIXME: I'm never sure if logand should be && or &. I think it currently is wrong. But I don't want to touch it because that would break existing code.
			     ;; 15
			     (:op (logior) :assoc l)
			     ;; 16
			     (:op (? throw co_yield setf =
				     incf decf
				     *= /= %=
				     <<= >>=
				     &= ^-	; |=
				     ) :assoc r)
			     (:op (comma) :assoc l)
			     )
  "This variable contains the C++ operator precedence table. 
   It is used in the PAREN* form to determine whether placing parentheses is necessary.")


(defun lookup-precedence (operator )
  "This function looks up the precedence of an operator in the precedence table."
  (loop for e in *precedence*
	and e-i from 0
	do
	   (destructuring-bind (&key op assoc) e
	    (when (member operator op)
	      (return e-i)))))

(defun lookup-associativity (operator )
	"This function looks up the associativity of an operator in the precedence table."
  (loop for e in *precedence*
	do
	   (destructuring-bind (&key op (assoc 'l)) e
	     (when (member operator op)
	       (return assoc)))))

;; The `string-op` class is used in the `emit-c` function for the
;; implementation of the PAREN* form to expand branches of the
;; abstract syntax tree into strings.  The `string-op` class
;; represents a string that also remembers the most recent operator
;; used. This operator can be used for lookup and precedence
;; comparison with the next operator.

(defclass string-op ()
  ((string :accessor string-of
	   :initarg :string
	   :initform (error "Must supply a string."))
   (operator :accessor operator-of
	     :initarg :operator
	     :initform (error "Must supply an operator."))))

(defmethod print-object ((object string-op) stream)
"Prints the string representation of STRING-OP object to a stream."
  (format stream "~a" (string-of object)))

(defun m (op str)
	"Create a STRING-OP object.

	This function creates an instance of the STRING-OP class with
	the given operator and string values. The purpose of this
	function is to conveniently create multiple instances of
	STRING-OP objects with a concise function name.

	Parameters:
		- op: The operator value for the STRING-OP object.
		- str: The string value for the STRING-OP object.

	Returns:
		A new instance of the STRING-OP class."

	(make-instance 'string-op
		 :string str
		 :operator op))


(defun m-of (obj-or-string)
  "This function is a shortcut to convert all types that can occur inside
emit-c into a string. Except lists: Those stay lists."
  (typecase obj-or-string
    (string-op
     (string-of obj-or-string))
    (string
     obj-or-string)
    (cons
     obj-or-string)
    (symbol
     (format nil "~a" obj-or-string))
    (t
     (break "variable '~a' of unknown type '~a'" obj-or-string (type-of obj-or-string)))))

(progn
  (defun emit-c (&key code (str nil)  (level 0) (hook-defun nil) (hook-defclass)
		   (current-class nil) (header-only nil) (in-class nil) (diag nil)
		   (omit-redundant-parentheses nil))

    "Evaluates s-expressions in CODE and emits a string or STRING-OP class.
     If HOOK-DEFUN is not nil, it calls hook-defun with every function definition.
     This functionality is intended to collect function declarations.
     When omit-redundant-parentheses is not nil, the feature to avoid redundant parentheses is active.

     Args:
     - code: The code as s-expressions to emit as C++.
     - str: A string to write the result into.
     - level: The indentation level.
     - hook-defun: The function to call with every function definition.
     - hook-defclass: The function to call with every class definition.
     - current-class: The current class.
     - header-only: A flag indicating whether to emit only the header. This flag can is used to emit only function declarations.
     - in-class: A flag indicating whether we are currently inside a class. This flag is used to emit method declarations or implementations. 
     - diag: If true, write diagnostic information as log output.
     - omit-redundant-parentheses: A flag indicating whether to avoid redundant parentheses.

     Returns:
     - The emitted string or string-op class."	   
 
					;(format t "~a~%" code)
					;(format t "header-only=~a~%" header-only)
    (flet ((emit (code &key (dl 0) (class current-class) (header-only-p header-only) (hook-fun hook-defun)
			 (hook-class hook-defclass) (in-class-p in-class) (current-diag diag)
			 (current-omit-redundant-parentheses omit-redundant-parentheses))
	     "change the indentation level. this is used in do"
	     (emit-c :code code
		     :level (+ dl level)
		     :hook-defun hook-fun
		     :hook-defclass hook-class
		     :current-class class
		     :header-only header-only-p
		     :in-class in-class-p
		     :diag current-diag
		     :omit-redundant-parentheses current-omit-redundant-parentheses)))
					;(format t "<emit-c type='~a' code='~a'>~%" (type-of code) code)
      (if code
	  (if (listp code)
	      (progn
		(case (car code)
		  (comma
		   ;; comma {args}*
		   (let ((args (cdr code)))
		     (m 'comma 
			(format nil "~{~a~^, ~}" (mapcar #'emit args)))))
		  (semicolon
		   ;; semicolon {args}*
		   (let ((args (cdr code)))
		     (m 'semicolon (format nil "~{~a~^; ~}" (mapcar #'emit args)))))

		  (scope
		   ;; scope {args}*
		   ;; double colons between arguments
		   (let ((args (cdr code)))
		     (m 'scope (format nil "~{~a~^::~}" (mapcar #'emit args)))))
		  
		  (space
		   ;; space {args}*
		   (let ((args (cdr code)))
		     (make-instance 'string-op
				    :string (format nil "~{~a~^ ~}" (mapcar #'emit args))
				    :operator 'space)))
		  (space-n
		   ;; space-n {args}*
		   ;; without semicolons
		   (let ((args (cdr code)))
		     (make-instance 'string-op
				    :string (format nil "~{~a~^ ~}" (mapcar #'emit args))
				    :operator 'space-n)))
		  (comments (let ((args (cdr code)))
			      (make-instance 'string-op
					     :string (format nil "~{// ~a~^~%~}" args)
					     :operator 'comments)))
		  (lines (let ((args (cdr code)))
			   ;; like comments but without the //
			   (make-instance 'string-op
					  :string (format nil "~{~a~%~}" args)
					  :operator 'lines)
			   ))
		  (doc ;; java doc comments
		   (let ((args (cdr code)))
		     (m 'doc
			#+nil
			(format nil "~a"
				(emit
				 `(do0
				   ,(format nil "/** ~a~%" (first args))
				   ,@(loop for line in (rest args)
					   collect
					   (format nil "* ~a~%" line))
				   ,(format nil "*/"))))
			(format nil "~{~a~}"
				`(
				  ,(format nil "/** ~a~%" (first args))
				  ,@(loop for line in (rest args)
					  collect
					  (format nil "* ~a~%" line))
				  ,(format nil "*/")))))
		   )
		  (paren*
		   ;; paren* parent-op arg
		   ;; place a pair of parentheses only when needed
		   ;; if omit-redundant-parentheses=true, act the same as paren

		   ;; The paren* form conditionally adds parentheses
		   ;; around an expression. It takes two arguments: a
		   ;; parent operator and an argument. The parent
		   ;; operator is the operator that is being applied
		   ;; to the argument. The argument can be any
		   ;; expression.

		   ;; The paren* form will add parentheses around the
		   ;; argument if the parent operator has a lower
		   ;; precedence in Python than the argument. This is
		   ;; necessary to ensure that the Python expression
		   ;; is evaluated in the correct order.
                       
		   
		   (if (not omit-redundant-parentheses)
		       (destructuring-bind (parent-op &rest args) (cdr code)
			 (m 'paren
			    (format nil "~a" (emit-c :code `(paren ,@args)))))
		       (progn  
					;(format t "<paren* code='~a'>~%" code)
			 (unless (eq 3 (length code))
			   (break "paren* expects only two arguments"))
			 (destructuring-bind (parent-op arg &rest rest) (cdr code)
					;let ((arg (second (cdr code))))
			   (cond
			     ((symbolp arg)
			      ;; no parens for symbol needed
			      (m 'symbol
				 (format nil (if diag "Asymbol.~a" "~a") (emit-c :code arg))))
			     ((numberp arg)
			      ;; no parens for number needed (maybe for negative?)
			      (m 'number
				 #+nil (format nil (if diag "Anumber.~a" "~a") (emit-c :code arg))
				 (if (<= 0 arg)
				     (format nil (if diag "Anumber.~a" "~a") (emit-c :code arg))
				     (format nil (if diag "Anegnumber.~a" " ~a") (emit-c :code arg)))))
			     ((stringp arg)
			      ;; no parens around string
			      (m 'string
				 (format nil (if diag  "Astring.~a" "~a") arg))
			      #+nil (progn
				      ;; a string may contain operators
				      ;; only add parens if there are not already parens

				      (if (and (eq #\( (aref arg 0))
					       (eq #\) (aref arg (- (length arg)
								    1))))
					  (format nil "~a" arg)
					  (format nil "_~a_" arg))))
			     ((and (typep arg 'string-op) (eq (operator-of arg) 'symbol))
			      ;; no parens for symbol needed
			      (make-instance 'string-op
					     :string (format nil (if diag "Asymbol.~a" "~a") arg)
					     :operator 'symbol))
			     ((and (typep arg 'string-op) (eq (operator-of arg) 'string))
			      ;; no parens around string
			      (m 'string
				 (format nil (if diag  "Astring.~a" "~a") arg)))
			     ((listp arg)
			      ;; a list can be an arbitrary abstract syntax tree of operators
			      (cond
				((<= (length arg) 2)
				 ;; two or one elements doesn't need paren
				 (let ((op0 (car arg)) 
				       (rest (cdr arg)))
				   (assert (or (symbolp op0)
					       (stringp op0)))
				   (assert (listp rest))
				   (emit (if diag
					     `(space ,(format nil "/*<~a len=~a>*/" arg (length arg) )
						     (,op0 ,@rest))
					     `(,op0 ,@rest)))))
				(t
				 (let ((op0 parent-op
					;(car arg)
					    ) ;; use precedence list to check if parens are needed
				       (rest  ; arg
					 (cdr arg)
					 ))
				   (assert (or (symbolp op0)
					       (stringp op0)))
				   (assert (listp rest))
				   (if (and (member  op0
						     *operators*)
					    (member  (car arg)
						     *operators*))
				       (let* ((p0 (lookup-precedence op0))
					      (p0assoc (lookup-associativity op0))
					      (op1 (car arg))
					      (p1 (lookup-precedence op1)
					;(+ 1 (length *precedence*))
						  )
					      (p1assoc ;'l
						(lookup-associativity op1)
						)
					      )
					 #+nil (loop for e in rest
						     do
							(when (or (listp e)
								  (typep e 'string-op))
							  (let* ((op1v (cond ((listp e) (first e))
									     ((typep e 'string-op) (operator-of e))
									     (t (break "unknown operator '~a'" e))))
								 
								 (p1v (lookup-precedence op1v))
								 (p1a (lookup-associativity op1v)))
							    (when p1
							      (setf op1 op1v
								    p1 p1v
								    p1assoc p1a)
							      ))))
					 ;; <paren* op0=hex p0=0 p1=18 rest=(ad) type=cons>
					 ;; (format t "<paren* op0=~a p0=~a p1=~a rest=~a type=~a>~%" op0 p0 p1 rest (type-of rest))
					 (if 
					  (or (< p0 p1)
					      (and (eq p0 p1)
						   (not (eq p0assoc p1assoc))
						   )
					      (member op0 `(/ % -))
					      (member op1 `(/ % -)))
					  (emit `(paren  ,(if diag
							      `(space ,(format nil "/*(op0='~a' op1='~a' arg=~a ~a)*/" op0 op1 arg (list  p0 p1 p0assoc p1assoc))
								      (,op1 ,@rest))
							      `(,op1 ,@rest))))
					  (emit (if diag
						    `(space ,(format nil "/*nopar op0='~a' (~a) op1='~a' arg=~a ~a*/" op0 (type-of op0) op1 arg (list  p0 p1 p0assoc p1assoc))
							    (,op1 ,@rest))
						    `(,op1 ,@rest)))))
				       (progn
					 ;; (break "unknown operator '~a'" op0)
					 ;; function call
					 (emit `(,(car arg) ,@rest))
					 ))))))
			     ((typep arg 'string-op)
			      (break "string-op ~a" arg)
			      arg	;(string-of arg)
			      )
			     (t
			      (break "unsupported argument for paren* '~a' type='~a'" arg (type-of arg))))))))
		  (paren
		   ;; paren {args}*
		   ;; parentheses with comma separated values
		   (let ((args (cdr code)))
		     (m 'paren
			(format nil "(~{~a~^, ~})" (mapcar #'emit args)))))
		  (angle
		   (let ((args (cdr code)))
		     (m 'angle
			(format nil "<~{~a~^, ~}>" (mapcar #'emit args)))))
		  (bracket
		   ;; bracket {args}*
		   (let ((args (cdr code)))
		     (m 'bracket
			(format nil "[~{~a~^, ~}]" (mapcar #'emit args)))))
		  (curly
		   ;; curly {args}*
		   (let ((args (cdr code)))
		     (m 'curly
			(format nil "{~{~a~^, ~}}" (mapcar #'emit args)))))
		  (designated-initializer
		   ;; designated-initializer {key val}*
		   ;; (designated-initializer Width Dimensions.Width Height Dimensions.Height)
		   ;; => {.Width = Dimensions.Width, .Height = Dimensions.Height}
					    
		   (let* ((args (cdr code)))
		     (m 'designated-initializer
			(format nil "~a"
				(emit `(curly ,@(loop for (e f) on args by #'cddr
						      collect
						      (if (symbolp e)
							  `(= ,(format nil ".~a" e) ,f)
							  `(= ,(format nil "~a" (emit e)) ,f)))))))))
		  (new
		   ;; new arg
		   (let ((arg (cadr code)))
		     (m 'new
			(format nil "new ~a" (emit arg)))))
		  (indent
		   ;; indent form
		   (m 'string
		      (format nil "~{~a~}~a"
			      ;; print indentation characters
			      (loop for i below level collect "    ")
			      (emit (cadr code)))))
		  (split-header-and-code
		   (let ((args (cdr code)))
		     (destructuring-bind (arg0 arg1) args
		       (if hook-defclass
			   (funcall hook-defclass (format nil "~a" (emit `(do0 ,arg0))))
			   (format nil "~a" (emit `(do0 ,arg1))))

		       )))
		  (do0 (m 'do0
			  (with-output-to-string (s)
			    ;; do0 {form}*
			    ;; write each form into a newline, keep current indentation level
					;(format t "<do0 type='~a' code='~a'>~%" (type-of (cadr code)) (cadr code) )
			    (format s "~{~&~a~}"
				    (mapcar
				     #'(lambda (xx)
					 (let* ((x (m-of xx))
						(bx (emit `(indent ,x) :dl 0))
						(b (m-of bx))
						(semicolon-maybe
						  (cond
						    ((typep b 'sequence)
						     (cond
						       ((eq #\; (aref b (- (length b) 1)))
							;; don't place second semicolon
							" " ; ".1."
							)
						       ((eq (type-of xx) 'string) ;(typep x 'string)
							;; don't place semicolon after strings
						        ".2."
							)
						       ((typep xx '(array character (*)))
							;; don't place semicolon after array of characters
							"" ; ".3."
							)
						       ((and (listp x)
							     (member (car x) `(defun do do0 progn
										for for-range dotimes
										while
										include include<> case
										when if unless cond
										let letc letd pragma
										split-header-and-code
										defun defun* defmethod defclass
										comments comment doc
										namespace
										handler-case
										space-n)))
							;; don't place semicolon after specific operators
							" " ;; ".4."
							)
						       ((eq #\Newline (aref b (- (length b) 1)))
							;; don't place semicolon after newline
							#\Newline ;; ".5."
							)
						       (t
							";" ;; ".6."
							)))
						    
						    ((not (typep b 'sequence))
						     (break "not a sequence type='~a' variable='~a'" (type-of b) b1))
						    (t ";"))))
					   #+nil (format t "<int-do0 type='~a' code='~a' b='~a' semicolon='~a'>~%"
							 (type-of xx) xx b semicolon-maybe)
					   
					   (format nil "~a~a"
						   b
						   ;; don't add semicolon if there is already one
						   ;; or if x contains a string
						   ;; or if x is an s-expression with a c thing that doesn't end with semicolon
						   semicolon-maybe)))
				     (cdr code)))
					; (terpri s)
					;(format t "</do0>~%")

			    #+nil
			    (let ((a (emit (cadr code))))
			      (format s "~&~a~a~{~&~a~}"
				      a
				      (if (eq #\; (aref a (- (length a) 1)))
					  ""
					  ";")
				      (mapcar
				       #'(lambda (x)
					   (let ((b (emit `(indent ,x) 0)))
					     (format nil "~a~a"
						     b
						     (if (eq #\; (aref b (- (length b) 1)))
							 ""
							 ";"))))
				       (cddr code)))))))
		  (pragma (m 'pragma
			     (let ((args (cdr code)))
			       (m 'pragma
				  (format nil "#pragma ~{~a~^ ~}" args)))))
		  (include (m 'include
			      (let ((args (cdr code)))
				(when (null args)
				  (break "no arguments in include"))
				;; include {name}*
				;; (include <stdio.h>)   => #include <stdio.h>
				;; (include interface.h) => #include "interface.h"
				(let ((str (with-output-to-string (s)
					     (loop for e in args do
					       ;; emit string if first character is not <
					       (format s "~&#include ~a"
						       (emit (if (eq #\< (aref (format nil "~a" e) 0))
								 e
								 `(string ,e))))))))
				  (when hook-defclass
				    (funcall hook-defclass (format nil "~a~%" str)))
				  str))))
		  (include<> (m 'include
				(let ((args (cdr code)))
				  ;; include {name}*
				  ;; (include<> stdio.h stddef.h)   => #include <stdio.h>\n #include<stddef.h>

				  (when (null args)
				    (break "no arguments in include<>"))
				  (let ((str (with-output-to-string (s)
					       (loop for e in args do

						 (format s "~&#include <~a>"
							 (emit e))))))
				    (when hook-defclass
				      (funcall hook-defclass (format nil "~a~%" str)))
				    str))))
		  (progn (with-output-to-string (s)
			   ;; progn {form}*
			   ;; like do but surrounds forms with braces.
			   (format s "{~{~&~a~}~&}" (mapcar #'(lambda (x) (emit `(indent (do0 ,x)) :dl 1)) (cdr code)))))
		  (namespace
		   ;; namespace name {form}*
		   ;; name .. can be nil
		   (let ((args (cdr code)))
		     (with-output-to-string (s)
		       (destructuring-bind (name &rest forms) args
			 (format s "namespace ~a ~a"
				 (emit name)
				 (emit `(progn ,@forms)))))))
		  (do (with-output-to-string (s)
			;; do {form}*
			;; print each form on a new line with one more indentation.
			(format s "~{~&~a~}" (mapcar #'(lambda (x) (emit `(indent (do0 ,x)) :dl 1)) (cdr code)))))
		  (defclass+
		   ;; for writing directly into header with all code in class
		   (destructuring-bind (name parents &rest body) (cdr code)
		     (let ((class-name (if (listp name)
					   (car name)
					   name))
			   (class-template nil)
			   (class-template-instance nil))
		       (when (listp name)
			 (destructuring-bind (name &key (template nil) (template-instance nil)) name
			   (setf class-name name
				 class-template template
				 class-template-instance template-instance)))
		       (format nil "~@[template<~a> ~]class ~a~@[<~a>~] ~@[: ~a~] ~a"

			       class-template
			       (emit class-name)

			       class-template-instance

			       (when parents
				 (emit `(comma ,parents)))
			       (emit `(progn ,@body)
				     :class nil ;(emit name)
				     :hook-fun nil
				     :hook-class hook-defclass
				     :header-only-p nil
				     :in-class-p 'defclass+))))
		   )
		  (defclass
			;; defclass class-name ({superclass-name}*) ({slot-specifier}*) [[class-option]]
			;; class TA : public Faculty, public Student { ... }
			;; defclass (class-name :template "T") ({superclass-name}*) ({slot-specifier}*) [[class-option]]
			;; template<T> class TA...
			;; defclass (class-name :template "T,y" :template-instance "std::integral_constant<T,y>") ...
			;; template<typename T, T y> class NameExtractor<std::integral_constant<T, y>>  {
			;; FIXME template stuff doesn't get emitted into cpp
			(destructuring-bind (name parents &rest body) (cdr code)
			  (let ((class-name (if (listp name)
						(progn
					;(format t "template-class: ~a~%" name)
						  (car name))
						name))
				(class-template nil)
				(class-template-instance nil))
			    (when (listp name)
			      (destructuring-bind (name &key (template nil) (template-instance nil)) name
				(setf class-name name
				      class-template template
				      class-template-instance template-instance)))

			    (prog1
				(if hook-defclass
				    (progn
				      ;; create class declaration with function headers
				      (funcall hook-defclass
					       (format nil "~@[template<~a> ~]class ~a~@[<~a>~] ~@[: ~a~] ~a"

						       class-template
						       (emit class-name)

						       class-template-instance

						       (when parents
							 (emit `(comma ,parents)))
						       (emit `(progn ,@body)
							     :class nil ;(emit name)
							     :hook-fun nil
							     :hook-class hook-defclass
							     :header-only-p t
					;:in-class-p 'defclass-cpp
							     )))
				      " ")
				    (progn
				      ;; only create function definitions of the class
				      ;; expand defun but non of the other commands
				      (destructuring-bind (name parents &rest body) (cdr code)
					(declare (ignorable parents))
					(with-output-to-string (s)
					  (loop for e in body do
					    (when (and (listp e)
						       (or (eq (car e) 'defmethod)
							   (eq (car e) 'defmethod*)))
					      (format s "~@[template< ~a > ~]~a"
						      class-template
						      (emit e
							    :class
							    (emit (format nil "~a~@[<~a>~]" class-name class-template-instance))
							    :header-only-p nil
							    :in-class-p 'defclass-cpp))))))))))))
		  (protected (format nil "protected ~a" (emit (cadr code))))
		  (public (format nil "public ~a" (emit (cadr code))))
		  (defmethod
		      (parse-defmethod code #'emit :class current-class :header-only header-only :in-class-p in-class))
		  #+nil (defmethod*
			 (if hook-defclass
			     (parse-defmethod code #'emit :class current-class :header-only t)
			     (parse-defmethod code #'emit :class current-class :header-only nil)))
		  (defun
		      (prog1
			  (parse-defun code #'emit :class current-class :header-only header-only)
					;(format t "defun ~a~%" (subseq code 0 (min 4 (length code))))
			(when hook-defun ;(and hook-defun (not current-class))
			  ;; only emit function headers when we are not currently in defclass
			  (funcall hook-defun (parse-defun code #'emit :header-only t :class current-class)))))
		  (defun* (parse-defun code #'emit :header-only t :class current-class))
		  (defun+ (parse-defun code #'emit :header-only nil :class current-class))
		  (return (m 'return
			     (format nil "return ~a" (emit (car (cdr code))))))
		  (co_return (m 'co_return
				(format nil "co_return ~a" (emit (car (cdr code))))))
		  (co_await (m 'co_await
			       (format nil "co_await ~a" (emit (car (cdr code))))))
		  (co_yield (m 'co_yield (format nil "co_yield ~a" (emit (car (cdr code))))))

		  (throw (m 'throw (format nil "throw ~a" (emit (car (cdr code))))))
		  (cast (destructuring-bind (type value) (cdr code)
			  (m 'cast
			     (format nil "(~a) ~a"
				     (emit type)
				     (emit value)))))

		  (let (parse-let code #'emit)) ;; normal variable declaration (defaults to auto), e.g. auto v =0;
		  (letc (parse-let code #'emit :const t)) ;; const declaration, e.g. const auto b = True;
		  (letd (parse-let code #'emit :decltype t)) ;; variable declaration with decltype, no definition; e.g. decltype(high_resolution_timer::now()) t0;
		  (setf
		   (let ((args (cdr code)))
		     ;; "setf {pair}*"
		     ;; (setf var 5) => var = 5;
		     ;; (setf a 5 b 4) => a=5; b=4;
		     (m 'setf
			(format nil "~a"
				(emit
				 `(do0
				   ,@(loop for i below (length args) by 2
					   collect
					   (let ((a (elt args i))
						 (b (elt args (+ 1 i))))
					     `(= ,a ,b)))))))))
		  (using
		   (let ((args (cdr code)))
		     ;; "using {pair}*"
		     ;; similar to setf
		     ;; (using var 5) => using var = 5;
		     ;; (using a 5 b 4) => using a=5; using b=4;
		     (format nil "~a"
			     (emit
			      `(do0
				,@(loop for i below (length args) by 2
					collect
					(let ((a (elt args i))
					      (b (elt args (+ 1 i))))
					  `(space using (= ,a ,b)))))))))
		  (not (m 'not (format nil "!~a" (emit `(paren* not ,(car (cdr code)))))))
		  (bitwise-not (m 'bitwise-not (format nil "~~~a" (emit `(paren* bitwise-not ,(cadr code))))))
		  (deref (m 'deref (format nil "*~a" (emit `(paren* deref ,(car (cdr code)))))))
		  (ref (m 'ref (format nil "&~a" (emit `(paren* ref ,(car (cdr code)))))))
		  (+ (let ((args (cdr code)))
		       ;; + {summands}*
		       (m '+
			  (format nil "~{~a~^+~}"

				  (mapcar
				   #'(lambda (x) (emit `(paren* + ,x)))
				   args)))))
		  (- (let ((args (cdr code)))
		       (if (eq 1 (length args))
			   (m '-unary (format nil " -~a" (emit `(paren* -unary ,(car args))))) ;; py
			   (m '- (format nil "~{~a~^-~}" (mapcar #'(lambda (x) (emit `(paren* - ,x))) args))))))
		  (* (m '*
			(let ((args (cdr code)))
			  (format nil "~{~a~^*~}" (mapcar #'(lambda (x) (emit `(paren* * ,x))) args)))))
		  (^ (m '^ (let ((args (cdr code)))
			     (format nil "~{~a~^^~}" (mapcar #'(lambda (x) (emit `(paren* ^ ,x))) args)))))
		  (xor `(^ ,@(cdr code)))
		  (& (m '& (let ((args (cdr code)))
			     (format nil "(~{~a~^&~})" (mapcar #'(lambda (x) (emit `(paren* & ,x))) args)))))
		  (/ (m '/ (let ((args (cdr code)))
			     (if (eq 1 (length args))
				 (format nil "1.0/~a" (emit `(paren* ,(car args)))) ;; py
				 (format nil "~{~a~^/~}" (mapcar #'(lambda (x) (emit `(paren* / ,x))) args))))))
		  (or (let ((args (cdr code))) ;; py
			(m 'or
			   (format nil "~{~a~^ | ~}" (mapcar #'(lambda (x) (emit `(paren* or ,x))) args)))))
		  (and (m 'and (let ((args (cdr code))) ;; py
				 (format nil "~{~a~^ & ~}" (mapcar #'(lambda (x) (emit `(paren* and ,x))) args)))))
		  #+nil (xor (let ((args (cdr code))) ;; py
			       (format nil "(~{(~a)~^ ^ ~})" (mapcar #'emit args))))
		  (logior (m 'logior (let ((args (cdr code)))
				       (format nil "~{~a~^||~}" (mapcar #'(lambda (x) (emit `(paren* logior ,x))) args)))))
		  (logand (m 'logand (let ((args (cdr code)))
				       (format nil "~{~a~^&&~}" (mapcar #'(lambda (x) (emit `(paren* logand ,x))) args)))))
		  (= (m '= (destructuring-bind (a b) (cdr code)
			     ;; = pair
			     (format nil "~a=~a" (emit `(paren* = ,a)) (emit `(paren* = ,b))))))
		  (/= (m '/= (destructuring-bind (a b) (cdr code)
			       (format nil "~a/=~a" (emit `(paren* /= ,a)) (emit `(paren* /= ,b))))))
		  (*= (m '*= (destructuring-bind (a b) (cdr code)
			       (format nil "~a*=~a" (emit `(paren* *= ,a)) (emit `(paren* *= ,b))))))
		  (^= (m '^= (destructuring-bind (a b) (cdr code)
			       (format nil "~a^=~a" (emit `(paren* ^= ,a)) (emit `(paren* ^= ,b))))))
		  (<=> (m '<=> (destructuring-bind (a b) (cdr code)
				 (format nil "~a<=>~a" (emit `(paren* <=> ,a)) (emit `(paren* <=> ,b))))))
		  (<= (m '<= (destructuring-bind (a b &optional c) (cdr code)
			       (if c
				   (format nil "~a<=~a && ~a<=~a"
					   (emit `(paren* ,a)) (emit `(paren* ,b))
					   (emit `(paren* ,b)) (emit `(paren* ,c)))
				   (format nil "~a<=~a" (emit `(paren* <= ,a)) (emit `(paren*  <= ,b)))))))
		  (< (m '< (destructuring-bind (a b &optional c) (cdr code)
			     (if c
				 (format nil "~a<~a && ~a<~a"
					 (emit `(paren* < ,a)) (emit `(paren* < ,b))
					 (emit `(paren* < ,b)) (emit `(paren* < ,c)))
				 (format nil "~a<~a"
					 (emit `(paren* < ,a))
					 (emit `(paren* < ,b)))))))
		  (!= (m '!= (destructuring-bind (a b) (cdr code)
			       (format nil "~a!=~a" (emit `(paren* != ,a)) (emit `(paren* != ,b))))))
		  (== (m '== (destructuring-bind (a b) (cdr code)
			       (format nil "~a==~a" (emit `(paren* == ,a)) (emit `(paren* == ,b))))))
		  
		  (% (m '% (destructuring-bind (a b) (cdr code)
			     (format nil "~a%~a" (emit `(paren* % ,a)) (emit `(paren* % ,b))))))
		  (<< (m '<< (destructuring-bind (a &rest rest) (cdr code)
			       (format nil "~a~{<<~a~}"
				       (emit `(paren* << ,a))
				       (mapcar #'(lambda (x) (emit `(paren* << ,x))) rest)))))
		  (>> (m '>> (destructuring-bind (a &rest rest) (cdr code)
			       (format nil "~a~{>>~a~}" (emit `(paren* >> ,a))
				       (mapcar #'(lambda (x) (emit `(paren* >> ,x))) rest)))))
		  (incf (m 'incf (destructuring-bind (a &optional b) (cdr code) ;; py
				   (if b
				       (format nil "~a+=~a" (emit `(paren* incf ,a))
					       (emit `(paren* incf ,b)))
				       (format nil "~a++" (emit `(paren* incf ,a)))))))
		  (decf (m 'decf (destructuring-bind (a &optional b) (cdr code)
				   (if b
				       (format nil "~a-=~a" (emit `(paren* decf ,a)) (emit `(paren* decf ,b)))
				       (format nil "~a--" (emit `(paren* decf ,a)))))))
		  (string (m 'string (format nil "\"~a\"" (cadr code))))
		  ;; if raw string contains )" it will stop, in order to prevent this a pre and suffix can be introduced, like R"x( .. )" .. )x"
		  (string-r (m 'string (format nil "R\"(~a)\"" (cadr code))))
		  (string-u8 (m 'string (format nil "u8\"(~a)\"" (cadr code))))
		  (char (m 'string (format nil "'~a'" (cadr code))))
		  (hex (m 'string (destructuring-bind (number) (cdr code)
				    (format nil "0x~x" number))))
		  (? (m '? (destructuring-bind (a b &optional c) (cdr code)
			     (if c
				 (format nil "~a ? ~a : ~a"
					 (emit `(paren* ? ,a))
					 ;; in C++ the second argument of ?: is interpreted as if it was placed in parens
					 (emit `(paren* paren ,b))
					 (emit `(paren* paren ,c)))
				 (format nil "~a ? ~a" (emit `(paren* ? ,a)) (emit `(paren* paren ,b)))))))
		  (if (destructuring-bind (condition true-statement &optional false-statement) (cdr code)
			(with-output-to-string (s)
			  (format s "if ( ~a ) ~a"
				  (emit condition)
				  (emit `(progn ,true-statement)))
			  (when false-statement
			    (format s " else ~a"
				    (emit `(progn ,false-statement)))))))
		  (if-constexpr (destructuring-bind (condition true-statement &optional false-statement) (cdr code)
			(with-output-to-string (s)
			  (format s "if constexpr ( ~a ) ~a"
				  (emit condition)
				  (emit `(progn ,true-statement)))
			  (when false-statement
			    (format s " else ~a"
				    (emit `(progn ,false-statement)))))))
		  (when (destructuring-bind (condition &rest forms) (cdr code)
			  (emit `(if ,condition
				     (do0
				      ,@forms)))))
		  (unless (destructuring-bind (condition &rest forms) (cdr code)
			    (emit `(if (not ,condition)
				       (do0
					,@forms)))))

		  (cond
		    ;; (cond (condition1 code1)
		    ;;       (condition2 code2)
		    ;;       (t          coden))
		    (with-output-to-string (s)
		      (destructuring-bind (&rest clauses) (cdr code)
			(loop for clause in clauses
			      and i from 0
			      do
				 (destructuring-bind (condition &rest expressions) clause
				   (if (equal condition t)
				       (format s "else ~a"
					       (emit `(progn ,@expressions)))
				       (format s "~a ( ~a ) ~a"
					       (if (eq i 0) 
						   "if"
						   "else if")
					       (emit condition)
					       (emit `(progn ,@expressions)))))))))

		  (dot (m 'dot
			  (let ((args (cdr code)))
			    (format nil "~{~a~^.~}" (mapcar #'emit (remove-if #'null
									      args))))))


		  (aref (m 'aref
			   (destructuring-bind (name &rest indices) (cdr code)
					;(format t "aref: ~a ~a~%" (emit name) (mapcar #'emit indices))
			     (format nil #-nil "~a~{[~a]~}"
					 #+nil "~a.at(~{~a~})" ;; I sometimes use this to debug (use a.at(i) instead of a[i] to have vector length check)
					 (emit `(paren* aref ,name))
				     ;; i think the second argument should feel like it is already surrounded by parens
				     (mapcar #'(lambda (x) (emit `(paren* paren ,x))) indices)))))

		  (-> (m '->
			 (let ((args (cdr code)))
			   (format nil "~{~a~^->~}" (mapcar #'(lambda (x) (emit `(paren* -> ,x))) args)))))

		  (lambda (parse-lambda code #'emit))

		  (case
		      ;; case keyform {normal-clause}* [otherwise-clause]
		      ;; normal-clause::= (keys form*)
		      ;; otherwise-clause::= (t form*)

		      (destructuring-bind (keyform &rest clauses)
			  (cdr code)
			(format
			 nil "switch ( ~a ) ~a"
			 (emit keyform)
			 (emit
			  `(progn
			     ,@(loop for c in clauses collect
						      (destructuring-bind (key &rest forms) c
							(if (eq key t)
							    (format nil "default: ~a"
								    (emit
								     `(progn
									,@forms #+nil (mapcar #'emit
											      forms)
									break)))
							    (format nil "case ~a: ~a"
								    (emit key)
								    (emit
								     `(progn
									,@forms #+nil (mapcar #'emit
											      forms)
									break)))))))))))
		  (for (destructuring-bind ((start end iter) &rest body) (cdr code)
			 (format nil "for ( ~@[~a~];~@[~a~];~@[~a~] ) ~a"
				 (emit start)
				 (emit end)
				 (emit iter)
				 (emit `(progn ,@body)))))
		  (for-range (destructuring-bind ((var range) &rest body)
				 (cdr code)
			       (multiple-value-bind (body env captures constructs const-p explicit-p inline-p static-p virtual-p noexcept-p final-p override-p pure-p template template-instance)
				   (consume-declare body)
				 (format str "for ( ~a ~a: ~a ) ~a"
					 (or (lookup-type var :env env)
					     ;"const auto&" ; *auto-keyword*
					     "auto&&"
					     ;; auto&& is treated as a forwarding reference by c++, variable declared with auto&& will result in an lvalue reference or an rvalue refrence depending on the value category of its initializer
					     )
					 (emit var)
					 (emit range)
					 (emit `(progn ,@body))))))
		  (dotimes (destructuring-bind ((i n &optional (step 1)) &rest body) (cdr code)
			     (multiple-value-bind (body env captures constructs const-p explicit-p inline-p static-p virtual-p noexcept-p final-p override-p pure-p template template-instance)
				 (consume-declare body)
			       (emit `(for (,(format nil "~a ~a = 0"

						     (or (lookup-type i :env env)
							 (emit `(decltype (+ 0 ,n 1))) ;*auto-keyword*
							 )

						     (emit i))
					    (< ,i ,n)
					    (incf ,i ,step))
					   ,@body)))))
		  #-generic-c
		  (foreach (destructuring-bind ((item collection) &rest body) (cdr code)
			     (format nil "for ( auto& ~a : ~a ) ~a"
				     (emit item)
				     (emit collection)
				     (emit `(progn ,@body)))))
		  #+generic-c
		  (foreach
		   (destructuring-bind ((item collection) &rest body) (cdr code)
		     (let ((itemidx (format nil "~a_idx" (emit item))))
		       (format nil
			       "~a"
			       (emit
				`(dotimes (,itemidx (/ (sizeof ,collection)
						       (sizeof (deref ,collection))))
				   (let ((,item (aref ,collection ,itemidx)))
				     (progn ,@body))))))))
		  (while ;; while condition {forms}*
		   (destructuring-bind (condition &rest body) (cdr code)
		     (format nil "while ( ~a ) ~a"
			     (emit condition)
			     (emit `(progn ,@body)))))
		  (deftype
		      ;; deftype name lambda-list {form}*
		      ;; only the first form of the body is used, lambda list is ignored
		      (destructuring-bind (name lambda-list &rest body) (cdr code)
			(declare (ignore lambda-list))
			(format nil "typedef ~a ~a" (emit (car body)) name)))
		  (struct (format nil "struct ~a" (emit (car (cdr code)))))
		  (defstruct0
		   ;; defstruct without init-form
		   ;; defstruct name {slot-description}*
		   ;; slot-description::= slot-name | (slot-name [slot-type])

		   ;; a slot-name without type can be used to create a
		   ;; composed type with a struct embedding

		   ;; i think i should use this pattern that works in C
		   ;; and in C++. Typedef isn't strictly necessary in
		   ;; C++, execept if you overload the struct name with
		   ;; a function:

		   ;; struct
		   ;; {
		   ;;    char name[50];
		   ;;    char street[100];
		   ;;    char city[50];
		   ;;    char state[20];
		   ;;    int pin;
		   ;; } Address;
		   ;; typedef struct Address Address;
		   ;; int Address(int b){ ...}

		   ;; https://stackoverflow.com/questions/1675351/typedef-struct-vs-struct-definitions
		   (destructuring-bind (name &rest slot-descriptions) (cdr code)
		     (format nil "~a"
			     (emit `(do0
				     ,(format nil "struct ~a ~a;"
					      name
					      (emit
					       `(progn
						  ,@(loop for desc in slot-descriptions collect
											(destructuring-bind (slot-name &optional type value) desc
											  (declare (ignorable value))
											  (format nil "~a ~a;" (emit type) (emit slot-name)))))))
				     #+generic-c (deftype ,name () (struct ,name)))))))
		  (handler-case
		      ;; handler-case expression [[{error-clause}*]]
;;; error-clause::= (typespec ([var]) declaration* form*) ;; note: declarations are currently unsupported
		      ;; error-clause::= (typespec ([var]) form*)
		      ;; if typespec is t, catch any kind of exception

		      ;; (handler-case (progn forma formb)
		      ;;   (typespec1 (var1) form1)
		      ;;   (typespec2 (var2) form2))

		      ;; a clause such as:
		      ;; (typespec (var) (declare (ignore var)) form)
		      ;; can be written as (typespec () form)



		      ;; try {
		      ;;   // code here
		      ;; }
		      ;; catch (int param) { cout << "int exception"; }
		      ;; catch (char param) { cout << "char exception"; }
		      ;; catch (...) { cout << "default exception"; }

		      (destructuring-bind (expr &rest clauses) (cdr code)
			(with-output-to-string (s)
			  (format s "try ~a"
				  (if (eq 'progn (car expr))
				      (emit expr)
				      (emit `(progn ,expr))))
			  (loop for clause in clauses do
			    (destructuring-bind (typespec (var) &rest forms) clause
			      (format s "catch (~a) ~a"
				      (if (and (eq 't typespec)
					       (null var))
					  (format nil "...")
					  (format nil "~a ~a" typespec var))
				      (emit `(progn ,@forms))))))))
		  (t (destructuring-bind (name &rest args) code

		       (if (listp name)
			   ;; lambda call and similar complex constructs
			   (format nil "(~a)~a"
				   (emit name)
				   (emit `(paren ,@args))
				   )
			   ;; function call


			   (progn	;if

			     #+nil(and
				   (= 1 (length args))
				   (eq (aref (format nil "~a" (car args)) 0) #\.))
			     #+nil (format nil "~a~a" name
					   (emit args))
			     (format nil "~a~a"
				     (emit name)
				     (emit `(paren ,@args)))))))))
	      
	      (cond
		((typep code 'string-op)
		 code)
		((symbolp code)
					;(cl-ppcre::regex-replace-all "--" "bla--fub" "::")
		 (m 'symbol (cl-ppcre::regex-replace-all "--" (format nil "~a" code) "::"))
					;(substitute #\: #\- (format nil "~a" code))
		 )
		((stringp code) ;; print variable
		 (m 'symbol (format nil "~a" code)))
		((numberp code) ;; print constants
		 (m 'number
		    (cond ((integerp code) (format str "~a" code))
			  ((floatp code)
			   (typecase code
			     (single-float (format str "~a" (print-sufficient-digits-f32 code)))
			     (double-float (format str "~a" (print-sufficient-digits-f64 code)))
			     )
			   #+nil (format str "(~a)" (print-sufficient-digits-f64 code))))))))
	  (m 'empty
	     ""))))
  #+nil (progn
	  (defparameter *bla*
	    (emit-c :code `(do0
			    (include <stdio.h>)
			    (defun main (argc argv)
			      (declare (type int argc)
				       (type char** argv)
				       (values int))
			      (printf (string "hello world!"))
			      (return 0)))))
	  (format t "~a" *bla*)))

#+nil((ntuple (let ((args (cdr code)))
		(format nil "~{~a~^, ~}" (mapcar #'emit args))))
      (paren
       ;; paren {args}*
       (let ((args (cdr code)))
	 (format nil "(~{~a~^, ~})" (mapcar #'emit args))))
      (braces
       ;; braces {args}*
       (let ((args (cdr code)))
	 (format nil "{~{~a~^, ~}}" (mapcar #'emit args))))
      (curly ;; name{arg1, args}
       ;; or name{key1:arg1, key2:arg2}
	     (destructuring-bind (name &rest args) (cdr code)
	       (emit `(cast ,name
			    (braces
			     ,@(if (keywordp (car args))
				   (loop for i below (length args) by 2 collect
									(let ((a (elt args i))
									      (b (elt args (+ 1 i))))
									  (format nil "~a: ~a" (emit a) (emit b))))
				   args))))))
      (cast ;; cast type value
	    (destructuring-bind (type value) (cdr code)
	      (format nil "~a ~a" (emit type) (emit value)))
	    )
      (dict
       ;; dict {pair}*
       (let* ((args (cdr code)))
	 (let ((str (with-output-to-string (s)
		      (loop for (e f) in args
			    do
			       (format s "~a: ~a," (emit e) (emit f))))))
	   (format nil "{~a}" ;; remobve trailing comma
		   (subseq str 0 (- (length str) 1))))))
      (go (format nil "go ~a" (emit (car (cdr code)))))
      (range (format nil "range ~a" (emit (car (cdr code)))))
      (chan (format nil "chan ~a" (emit (car (cdr code)))))
      (defer (format nil "defer ~a" (emit (car (cdr code)))))
      (return (format nil "return ~a" (emit (car (cdr code)))))

      (do (with-output-to-string (s)
	    ;; do {form}*
	    ;; print each form on a new line with one more indentation.
	    (format s "~{~&~a~}" (mapcar #'(lambda (x) (emit `(indent ,x) 1)) (cdr code)))
	    (progn (with-output-to-string (s)
		     ;; progrn {form}*
		     ;; like do but surrounds forms with braces.
		     (format s "{~{~&~a~}~&}" (mapcar #'(lambda (x) (emit `(indent ,x) 1)) (cdr code)))))))

      (let (parse-let code #'emit))

      (defun (parse-defun code #'emit))
      (defun-declaration (parse-defun-declaration code #'emit))

      (defmethod (parse-defmethod code #'emit))
      (defmethod-interface (parse-defmethod-interface code #'emit))
      (defmethod-declaration (parse-defmethod-declaration code #'emit))
      #+nil (defstruct
		;;  defstruct name {slot-description}*
		;; slot-description::= slot-name | (slot-name [slot-initform [[slot-option]]])
		;; slot-option::= :type slot-type
		(destructuring-bind (name &rest slot-descriptions) (cdr code)
		  (format
		   nil "type ~a struct ~a"
		   name
		   (emit
		    `(progn
		       ,@(loop for desc in slot-descriptions collect
							     (destructuring-bind (slot-name ;; &optional init
										  ;; init doesnt really fit into go semantics
										  &key type) desc
							       (format nil "~a~@[ ~a~]" slot-name type))))))))
      (deftype
	  ;; deftype name lambda-list {form}*
	  ;; only the first form of the body is used, lambda list is ignored
	  (destructuring-bind (name lambda-list &rest body) (cdr code)
	    (declare (ignore lambda-list))
	    (format nil "type ~a ~a" name (emit (car body)))))


      (definterface

	  ;; definterface name {slot-description}*
	  ;; slot-description::= other-interface-name | method-interface-declaration

	  (destructuring-bind (name &rest slot-descriptions) (cdr code)
	    (format nil "type ~a interface ~a"
		    name
		    (emit
		     `(progn
			,@(mapcar #'emit slot-descriptions))))))
      (setf (parse-setf code #'emit)) 
      (const (parse-const code #'emit))
      (assign
       ;; assign {pair}*
       (let ((args (cdr code)))
	 (format nil "~a~%"
		 (emit
		  `(do0
		    ,@(loop for i below (length args) by 2 collect
							   (let ((a (elt args i))
								 (b (elt args (+ 1 i))))
							     `(:= ,a ,b))))))))

      (ecase
	  ;; ecase keyform {normal-clause}*
	  ;; normal-clause::= (keys form*)
	  (destructuring-bind (keyform &rest clauses)
	      (cdr code)
	    (format
	     nil "switch ~a ~a"
	     (emit keyform)
	     (emit
	      `(progn
		 ,@(loop for c in clauses collect
					  (destructuring-bind (key &rest forms) c
					    (format nil "case ~a:~&~a"
						    (emit key)
						    (emit
						     `(do0
						       ,@(mapcar #'emit
								 forms)))))))))))

      (for
       ;; for [init [condition [update]]] {forms}*
       (destructuring-bind ((&optional init condition update) &rest body)
	   (cdr code)
	 (with-output-to-string (s)
	   (format s "for ~a ; ~a; ~a "
		   (if init
		       (emit init)
		       "")
		   (if condition
		       (emit condition)
		       "")
		   (if update
		       (emit update)
		       ""))
	   (format s "~a" (emit `(progn ,@body))))))
      (foreach
       ;; foreach [var] range {forms}*
       ;; foreach range {forms}*
       (destructuring-bind ((&rest decl) &rest body) (cdr code)
	 (with-output-to-string (s)
	   (format s "for ~a "
		   (if (< 1 (length decl))
		       (destructuring-bind (var range) decl
			 (emit `(:= ,var ,range)))
		       (emit (car decl))))
	   (format s "~a" (emit `(progn ,@body))))))


      (dotimes (destructuring-bind ((var end) &rest body) (cdr code)
		 (emit `(for ((:= ,var 0)
			      (< ,var ,end)
			      (incf ,var))
			     ,@body))))


      (slice (let ((args (cdr code)))
	       (if (null args)
		   (format nil ":")
		   (format nil "~{~a~^:~}" (mapcar #'emit args)))))

      #+nil (-> (let ((forms (cdr code)))
		  ;; clojure's thread first macro, thrush operator
		  ;; http://blog.fogus.me/2010/09/28/thrush-in-clojure-redux/
		  ;; -> {form}*
		  (emit (reduce #'(lambda (x y) (list (emit x) (emit y))) forms)))))
