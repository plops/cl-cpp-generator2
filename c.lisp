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
;(setf *features* (union *features* '(:generic-c)))
;(setf *features* (set-difference *features* '(:generic-c)))
(in-package :cl-cpp-generator2)

(setf (readtable-case *readtable*) :invert)

(defparameter *file-hashes* (make-hash-table))
(defparameter *auto-keyword* "auto")

(defun write-source (name code &key
				 (dir (user-homedir-pathname))
				 ignore-hash
				 (format t))
  (let* ((fn (merge-pathnames (format nil "~a" name)
			      dir))
	 (code-str (emit-c :code code :header-only nil))
	 (fn-hash (sxhash fn))
	 (code-hash (sxhash code-str)))
    (format t "write code into file: '~a'~%" fn)
    (multiple-value-bind (old-code-hash exists) (gethash fn-hash *file-hashes*)
      (when (or (not exists) ignore-hash (/= code-hash old-code-hash)
		(not (probe-file fn)))
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
	;; FIXME: figure out how to prevent that
       (when format
	 (sb-ext:run-program "/usr/bin/clang-format"
			     (list "-i"  (namestring fn)
				   ;; "-style='{PenaltyReturnTypeOnItsOwnLine: 100000000}'"
				   )))))))

;; http://clhs.lisp.se/Body/s_declar.htm
;; http://clhs.lisp.se/Body/d_type.htm

;; go through the body until no declare anymore



  ;; (declare (type int a b) (type float c) 
  ;; (declare (values int &optional))
  ;; (declare (values int float &optional))

  ;; FIXME doesnt handle documentation strings



;; FIXME: misses const override
(defun consume-declare (body)
  "take a list of instructions from body, parse type declarations,
return the body without them and a hash table with an environment. the
entry return-values contains a list of return values. currently supports type, values, construct and capture declarations. construct is member assignment in constructors. capture is for lambda functions"
  (let ((env (make-hash-table))
	(captures nil)
	(constructs nil)
	(const-p nil)
	(explicit-p nil)
	(inline-p nil)
	(static-p nil)
	(virtual-p nil)
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
			    (setf virtual-p t)
			    )
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
			    ;; only collect types until occurrance of &optional
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
	    const-p explicit-p inline-p static-p virtual-p template template-instance)))

(defun lookup-type (name &key env)
  "get the type of a variable from an environment"
  (gethash name env))

(defun variable-declaration (&key name env emit)
  (let* ((type (lookup-type name :env env)))
    (cond ((null type)

	   (format nil "~a ~a"
		   ;#+generic-c "__auto_type"
					; #-generic-c "auto"
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
		    ;#+generic-c "__auto_type"
					;#-generic-c "auto"
		    *auto-keyword*
		    )
		(funcall emit name))))
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
		(funcall emit name)))))

(defun parse-let (code emit)
  "let ({var | (var [init-form])}*) declaration* form*"
  (destructuring-bind (decls &rest body) (cdr code)
    (multiple-value-bind (body env captures constructs const-p explicit-p inline-p static-p virtual-p template template-instance) (consume-declare body)
      (with-output-to-string (s)
	(format s "~a"
		(funcall emit
			`(do0
			  ,@(loop for decl in decls collect
				 (if (listp decl) ;; split into name and initform
				     (destructuring-bind (name &optional value) decl
				       (format nil "~a ~@[ = ~a~];"
					       (variable-declaration :name name :env env :emit emit)
					       
					       (when value
						   (funcall emit value))))
				     (format nil "~a;"
					     (variable-declaration :name decl :env env :emit emit))))
			  ,@body)))))))

(defun parse-defun (code emit &key header-only (class nil))
  ;; defun function-name lambda-list [declaration*] form*
  (destructuring-bind (name lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env captures constructs const-p explicit-p inline-p static-p virtual-p template template-instance) (consume-declare body) ;; py
      (multiple-value-bind (req-param opt-param res-param
				      key-param other-key-p
				      aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(with-output-to-string (s)

	  
	  ;;         template          static          inline  virtual ret   params     header-only
	  ;;                                   explicit                   name  const             constructs
	  ;;         1                 2       3       4       5       6  7  8  9       10        11 
	  (format s "~@[template<~a> ~]~@[~a ~]~@[~a ~]~@[~a ~]~@[~a ~]~a ~a ~a ~@[~a~] ~:[~;;~]  ~@[: ~a~]"
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
		  ;; 9 const keyword
		  (when const-p #+nil
			(and const-p
			     (not header-only))
			"const")
		  
		  ;; 10 semicolon if header only
		  header-only
		  ;; 11 constructor initializers
		  (when (and constructs
			     (not header-only))
		    (funcall emit `(comma ,@(mapcar emit constructs)))))
	  (unless header-only
	    (format s "~a" (funcall emit `(progn ,@body)))))))))

(defun parse-defmethod (code emit &key header-only (class nil) (in-class-p nil))
  ;; defun function-name lambda-list [declaration*] form*
  (destructuring-bind (name lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env captures constructs const-p explicit-p inline-p static-p virtual-p template template-instance) (consume-declare body) ;; py
      (multiple-value-bind (req-param opt-param res-param
				      key-param other-key-p
				      aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(when (and inline-p (not header-only))
	  (return-from parse-defmethod ""))
	(with-output-to-string (s)
	  ;;         template          static          inline  virtual ret   params     header-only
	  ;;                                   explicit                   name  const             constructs
	  ;;         1                 2       3       4       5       6  7  8  9       10        11 
	 ; format s "~@[template<~a> ~]~@[~a ~]~@[~a ~]~@[~a ~]~@[~a ~]~a ~a ~a ~@[~a~] ~:[~;;~]  ~@[: ~a~]"
	
	  (format s "~@[template<~a> ~]~@[~a ~]~@[~a ~]~@[~a ~]~@[~a ~]~a ~a ~a ~@[~a~] ~:[~;;~]  ~@[: ~a~]"
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
		  
		  ;; semicolon if header only
		  (and (not inline-p) header-only)
		  ;; constructor initializers
		  (when (and constructs
			 (not header-only))
		    (funcall emit `(comma ,@(mapcar emit constructs)))))
	  (when (or inline-p (not header-only))
	   (format s "~a" (funcall emit `(progn ,@body)))))))))

(defun parse-lambda (code emit)
  ;;  lambda lambda-list [declaration*] form*
  ;; no return value:
  ;;  [] (int a, float b) { body }
  ;; with (declaration (values float)):
  ;;  [] (int a, float b) -> float { body }
  ;; support for captures (placed into the first set of brackets)
  ;; (declare (capture &app bla)) will result in [&app, bla]
  (destructuring-bind (lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env captures constructs const-p) (consume-declare body)
      (multiple-value-bind (req-param opt-param res-param
				      key-param other-key-p
				      aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(with-output-to-string (s)
	  (format s "[~{~a~^,~}] ~a~@[-> ~a ~]"
		  (mapcar emit captures)
		  (funcall emit `(paren
				  ,@(loop for p in req-param collect
					 (format nil "~a ~a"
						 (let ((type (gethash p env)))
						   (if type
						       (funcall emit type)
						       (break "can't find type for ~a in defun"
							      p)))
						 p
						 ))))
		  (let ((r (gethash 'return-values env)))
		    (if (< 1 (length r))
			(funcall emit `(paren ,@r))
			(car r))))
	  (format s "~a" (funcall emit `(progn ,@body))))))))

(defun print-sufficient-digits-f32 (f)
  "print a single floating point number as a string with a given nr. of                                                                                                                                             
  digits. parse it again and increase nr. of digits until the same bit                                                                                                                                              
  pattern."
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
	(format nil "~af" (string-trim '(#\Space) str))
	#+nil
	(if (find #\e str)
	    str
	    (format nil "~af" (string-trim '(#\Space) str))))))


(defun print-sufficient-digits-f64 (f)
  "print a double floating point number as a string with a given nr. of                                                                                                                                             
  digits. parse it again and increase nr. of digits until the same bit                                                                                                                                              
  pattern."

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

			  
(progn
  (defun emit-c (&key code (str nil)  (level 0) (hook-defun nil) (hook-defclass) (current-class nil) (header-only nil) (in-class nil))
    "evaluate s-expressions in code, emit a string. if hook-defun is not nil, hook-defun will be called with every function definition. this functionality is intended to collect function declarations."
					;(format t "~a~%" code)
    ;(format t "header-only=~a~%" header-only)
    (flet ((emit (code &key (dl 0) (class current-class) (header-only-p header-only) (hook-fun hook-defun)
			 (hook-class hook-defclass) (in-class-p in-class))
	     "change the indentation level. this is used in do"
	     (emit-c :code code
		     :level (+ dl level)
		     :hook-defun hook-fun
		     :hook-defclass hook-class
		     :current-class class
		     :header-only header-only-p
		     :in-class in-class-p)))
      (if code
	  (if (listp code)
	      (progn
		(case (car code)
		  (comma
		   ;; comma {args}*
		   (let ((args (cdr code)))
		     (format nil "~{~a~^, ~}" (mapcar #'emit args))))
		  (semicolon
		   ;; semicolon {args}*
		   (let ((args (cdr code)))
		     (format nil "~{~a~^; ~}" (mapcar #'emit args))))
		  (space
		   ;; space {args}*
		   (let ((args (cdr code)))
		     (format nil "~{~a~^ ~}" (mapcar #'emit args))))
		  (comments (let ((args (cdr code)))
                              (format nil "~{// ~a~%~}" args)))
		  (paren
		   ;; paren {args}*
		   (let ((args (cdr code)))
		     (format nil "(~{~a~^, ~})" (mapcar #'emit args))))
		  (angle
		   (let ((args (cdr code)))
		     (format nil "<~{~a~^, ~}>" (mapcar #'emit args))))
		  (bracket
		   ;; bracket {args}*
		   (let ((args (cdr code)))
		     (format nil "[~{~a~^, ~}]" (mapcar #'emit args))))
		  (curly
		   ;; curly {args}*
		   (let ((args (cdr code)))
		     (format nil "{~{~a~^, ~}}" (mapcar #'emit args))))
		  (new
		   ;; new arg
		   (let ((arg (cadr code)))
		     (format nil "new ~a" (emit arg))))
		  (indent
		   ;; indent form
		   (format nil "~{~a~}~a"
			   ;; print indentation characters
			   (loop for i below level collect "    ")
			   (emit (cadr code))))
		  (split-header-and-code
		   (let ((args (cdr code)))
		     (destructuring-bind (arg0 arg1) args
		       (if hook-defclass
			   (funcall hook-defclass (format nil "~a" (emit `(do0 ,arg0))))
			   (format nil "~a" (emit `(do0 ,arg1))))
		       
		       )))
		  (do0 (with-output-to-string (s)
			 ;; do0 {form}*
			 ;; write each form into a newline, keep current indentation level
			 (format s "~{~&~a~}"
				 (mapcar
				  #'(lambda (x)
				      (let ((b (emit `(indent ,x) :dl 0)))
					(format nil "~a~a"
						b
						;; don't add semicolon if there is already one
						;; or if x contains a string
						;; or if x is an s-expression with a c thing that doesn't end with semicolon
						(if (or (eq #\; (aref b (- (length b) 1)))
							(and (typep x 'string))
							(and (typep x '(array character (*))))
							(and (listp x)
							     (member (car x) `(defun do do0 progn
										for for-range dotimes
										while
										include case
										when if unless
										let
										split-header-and-code
										defun defmethod defclass))))
						    ""
						    ";"))))
				  (cdr code)))
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
				    (cddr code))))))
		  (include (let ((args (cdr code)))
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
			       str)))
		  (progn (with-output-to-string (s)
			   ;; progn {form}*
			   ;; like do but surrounds forms with braces.
			   (format s "{~{~&~a~}~&}" (mapcar #'(lambda (x) (emit `(indent (do0 ,x)) :dl 1)) (cdr code)))))
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
		  (return (format nil "return ~a" (emit (car (cdr code)))))
		  (co_return (format nil "co_return ~a" (emit (car (cdr code)))))
		  (co_await (format nil "co_await ~a" (emit (car (cdr code)))))
		  (co_yield (format nil "co_yield ~a" (emit (car (cdr code)))))
		  
		  (throw (format nil "throw ~a" (emit (car (cdr code)))))
		  (cast (destructuring-bind (type value) (cdr code)
			  (format nil "(~a) ~a"
				  (emit type)
				  (emit value))))
		  
		  (let (parse-let code #'emit))
		  (setf 
		   (let ((args (cdr code)))
		     ;; "setf {pair}*"
		     (format nil "~a"
			     (emit
			      `(do0 
				,@(loop for i below (length args) by 2 collect
								       (let ((a (elt args i))
									     (b (elt args (+ 1 i))))
									 `(= ,a ,b))))))))
		  (not (format nil "!(~a)" (emit (car (cdr code)))))
		  (deref (format nil "*(~a)" (emit (car (cdr code)))))
		  (ref (format nil "&(~a)" (emit (car (cdr code)))))
		  (+ (let ((args (cdr code)))
		       ;; + {summands}*
		       (format nil "(~{(~a)~^+~})" (mapcar #'emit args))))
		  (- (let ((args (cdr code)))
		       (if (eq 1 (length args))
			   (format nil "(-(~a))" (emit (car args))) ;; py
			   (format nil "(~{(~a)~^-~})" (mapcar #'emit args)))))
		  (* (let ((args (cdr code)))
		       (format nil "(~{(~a)~^*~})" (mapcar #'emit args))))
		  (^ (let ((args (cdr code)))
		       (format nil "(~{(~a)~^^~})" (mapcar #'emit args))))
		  (& (let ((args (cdr code)))
		       (format nil "(~{(~a)~^&~})" (mapcar #'emit args))))
		  (/ (let ((args (cdr code)))
		       (if (eq 1 (length args))
			   (format nil "(1.0/(~a))" (emit (car args))) ;; py
			   (format nil "(~{(~a)~^/~})" (mapcar #'emit args)))))
		  
		  (logior (let ((args (cdr code))) ;; py
			    (format nil "(~{(~a)~^ | ~})" (mapcar #'emit args))))
		  (logand (let ((args (cdr code))) ;; py
			    (format nil "(~{(~a)~^ & ~})" (mapcar #'emit args))))
		  (logxor (let ((args (cdr code))) ;; py
			    (format nil "(~{(~a)~^ ^ ~})" (mapcar #'emit args))))
		  (or (let ((args (cdr code)))
			(format nil "(~{(~a)~^||~})" (mapcar #'emit args))))
		  (and (let ((args (cdr code)))
			 (format nil "(~{(~a)~^&&~})" (mapcar #'emit args))))
		  (= (destructuring-bind (a b) (cdr code)
		       ;; = pair
		       (format nil "~a=~a" (emit a) (emit b))))
		  (/= (destructuring-bind (a b) (cdr code)
			(format nil "~a/=(~a)" (emit a) (emit b))))
		  (*= (destructuring-bind (a b) (cdr code)
			(format nil "~a*=(~a)" (emit a) (emit b))))
		  (^= (destructuring-bind (a b) (cdr code)
			(format nil "(~a)^=(~a)" (emit a) (emit b))))
		  (<= (destructuring-bind (a b &optional c) (cdr code)
			(if c
			    (format nil "(((~a)<=(~a)) && ((~a)<=(~a)))" (emit a) (emit b)
				    (emit b) (emit c))
			    (format nil "(~a)<=(~a)" (emit a) (emit b)))))
		  (< (destructuring-bind (a b &optional c) (cdr code)
		       (if c
			   (format nil "(((~a)<(~a)) && ((~a)<(~a)))" (emit a) (emit b)
				   (emit b) (emit c))
			   (format nil "(~a)<(~a)" (emit a) (emit b)))))
		  (!= (destructuring-bind (a b) (cdr code)
			(format nil "(~a)!=(~a)" (emit a) (emit b))))
		  (== (destructuring-bind (a b) (cdr code)
			(format nil "(~a)==(~a)" (emit a) (emit b))))
		  
		  (% (destructuring-bind (a b) (cdr code)
		       (format nil "~a%~a" (emit a) (emit b))))
		  (<< (destructuring-bind (a &rest rest) (cdr code)
			(format nil "(~a)~{<<(~a)~}" (emit a) (mapcar #'emit rest))))
		  (>> (destructuring-bind (a &rest rest) (cdr code)
			(format nil "(~a)~{>>(~a)~}" (emit a) (mapcar #'emit rest))))
		  #+nil (>> (destructuring-bind (a b) (cdr code)
			      (format nil "(~a)>>~a" (emit a) (emit b))))
		  (incf (destructuring-bind (a &optional b) (cdr code) ;; py
			  (if b
			      (format nil "(~a)+=(~a)" (emit a) (emit b))
			      (format nil "(~a)++" (emit a)))))
		  (decf (destructuring-bind (a &optional b) (cdr code)
			  (if b
			      (format nil "(~a)-=(~a)" (emit a) (emit b))
			      (format nil "(~a)--" (emit a)))))
		  (string (format nil "\"~a\"" (cadr code)))
		  (string-r (format nil "R\"(~a)\"" (cadr code)))
		  (string-u8 (format nil "u8\"(~a)\"" (cadr code)))
		  (char (format nil "'~a'" (cadr code)))
		  (hex (destructuring-bind (number) (cdr code)
			 (format nil "0x~x" number)))
		  (? (destructuring-bind (a b &optional c) (cdr code)
		       (if c
			   (format nil "(~a) ? (~a) : (~a)" (emit a) (emit b) (emit c))
			   (format nil "(~a) ? (~a)" (emit a) (emit b)))))
		  (if (destructuring-bind (condition true-statement &optional false-statement) (cdr code)
			(with-output-to-string (s)
			  (format s "if ( ~a ) ~a"
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
		  
		  (dot (let ((args (cdr code)))
			 (format nil "~{~a~^.~}" (mapcar #'emit args))))
		  

		  (aref (destructuring-bind (name &rest indices) (cdr code)
					;(format t "aref: ~a ~a~%" (emit name) (mapcar #'emit indices))
			  (format nil "~a~{[~a]~}" (emit name) (mapcar #'emit indices))))
		  
		  (-> (let ((args (cdr code)))
			(format nil "~{~a~^->~}" (mapcar #'emit args))))
		  
		  (lambda (parse-lambda code #'emit))
		  
		  (case
		      ;; case keyform {normal-clause}* [otherwise-clause]
		      ;; normal-clause::= (keys form*) 
		      ;; otherwise-clause::= (t form*) 
		      
		      (destructuring-bind (keyform &rest clauses)
			  (cdr code)
			(format
			 nil "switch(~a) ~a"
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
			 (format nil "for (~@[~a~];~@[~a~];~@[~a~]) ~a"
				 (emit start)
				 (emit end)
				 (emit iter)
				 (emit `(progn ,@body)))))
		  (for-range (destructuring-bind ((var-decl range) &rest statement-list)
				 (cdr code)
			       (format str "for(~a : ~a) ~a"
				       (if (atom var-decl)
					   (format nil "auto ~a" var-decl)
					   (destructuring-bind (name &key (type 'auto)) var-decl
					     (format nil "~a ~a" type name)))
				       (emit range)
				       (emit `(progn ,@statement-list)))))
		  (dotimes (destructuring-bind ((i n &optional (step 1)) &rest body) (cdr code)
			     (emit `(for (,(format nil "~a ~a = 0"
					;#+generic-c "__auto_type"
					;#-generic-c "auto"
						   *auto-keyword*
						   (emit i)) ;; int
					  (< ,(emit i) ,(emit n))
					  (incf ,(emit i) ,(emit step)))
					 ,@body))))
		  #-generic-c
		  (foreach (destructuring-bind ((item collection) &rest body) (cdr code)
			     (format nil "for (auto& ~a : ~a) ~a"
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
		     (format nil "while (~a) ~a"
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
				     (deftype ,name () (struct ,name)))))))
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
		((symbolp code)
					;(cl-ppcre::regex-replace-all "--" "bla--fub" "::")
		 (cl-ppcre::regex-replace-all "--" (format nil "~a" code) "::")
					;(substitute #\: #\- (format nil "~a" code))
		 )
		((stringp code) ;; print variable
		 (format nil "~a" code))
		((numberp code) ;; print constants
		 (cond ((integerp code) (format str "~a" code))
		       ((floatp code)
			(typecase code
			  (single-float (format str "(~a)" (print-sufficient-digits-f32 code)))
			  (double-float (format str "(~a)" (print-sufficient-digits-f64 code))))
			#+nil (format str "(~a)" (print-sufficient-digits-f64 code)))))))
	  "")))
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
