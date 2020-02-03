
#-nil
(progn (ql:quickload "alexandria")
       (defpackage :cl-cpp-generator2
	 (:use :cl
	       :alexandria)
	 (:export
	  #:write-source)))
;(setf *features* (union *features* '(:generic-c)))
;(setf *features* (set-difference *features* '(:generic-c)))
(in-package :cl-cpp-generator2)

(setf (readtable-case *readtable*) :invert)

(defparameter *file-hashes* (make-hash-table))

(defun write-source (name code &optional (dir (user-homedir-pathname))
				 ignore-hash)
  (let* ((fn (merge-pathnames (format nil "~a" name)
			      dir))
	(code-str (emit-c :code code))
	(fn-hash (sxhash fn))
	 (code-hash (sxhash code-str)))
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
	(sb-ext:run-program "/usr/bin/clang-format"
			    (list "-i"  (namestring fn)
					;"-style='{PenaltyReturnTypeOnItsOwnLine: 100000000}'"
				  ))))))

;; http://clhs.lisp.se/Body/s_declar.htm
;; http://clhs.lisp.se/Body/d_type.htm

;; go through the body until no declare anymore



  ;; (declare (type int a b) (type float c)
  ;; (declare (values int &optional))
  ;; (declare (values int float &optional))

  ;; FIXME doesnt handle documentation strings

(defun consume-declare (body)
  "take a list of instructions from body, parse type declarations,
return the body without them and a hash table with an environment. the
entry return-values contains a list of return values"
  (let ((env (make-hash-table))
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
		      (when (eq (first declaration) 'values)
			(destructuring-bind (symb &rest types-opt) declaration
			  (declare (ignorable symb))
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
    (values (reverse new-body) env)))

(defun lookup-type (name &key env)
  "get the type of a variable from an environment"
  (gethash name env))

(defun variable-declaration (&key name env emit)
  (let* ((type (lookup-type name :env env)))
    (if (listp type)
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
    (multiple-value-bind (body env) (consume-declare body)
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

(defun parse-defun (code emit &key header-only)
  ;; defun function-name lambda-list [declaration*] form*
  (destructuring-bind (name lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env) (consume-declare body) ;; py
      (multiple-value-bind (req-param opt-param res-param
				      key-param other-key-p
				      aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(with-output-to-string (s)
	  (format s "~a ~a ~a~:[~;;~]"
		  (let ((r (gethash 'return-values env)))
		    (if (< 1 (length r))
					;(funcall emit `(paren ,@r))
			(break "multiple return values unsupported: ~a"
			       r)
			(if (car r)
			    (car r)
			    "void")))
		  name
		  (funcall emit `(paren
				  ,@(loop for p in req-param collect
					 (format nil "~a ~a"
						 (let ((type (gethash p env)))
						   (if type
						       type
						       (break "can't find type for ~a in defun"
							      p)))
						 p))))
		  header-only)
	  (unless header-only
	   (format s "~a" (funcall emit `(progn ,@body)))))))))

(defun parse-lambda (code emit)
  ;;  lambda lambda-list [declaration*] form*
  ;; no return value:
  ;;  [] (int a, float b) { body }
  ;; with (declaration (values float)):
  ;;  [] (int a, float b) -> float { body }
  ;; currently no support for captures (which would be placed into the first set of brackets)
  (destructuring-bind (lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env) (consume-declare body)
      (multiple-value-bind (req-param opt-param res-param
				      key-param other-key-p
				      aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(with-output-to-string (s)
	  (format s "[] ~a~@[-> ~a ~]"
		  (funcall emit `(paren
				  ,@(loop for p in req-param collect
					 (format nil "~a ~a"
						 (let ((type (gethash p env)))
						   (if type
						       type
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
  (let* ((ff (coerce f 'single-float))
         (s (format nil "~E" ff)))
    #+nil   (assert (= 0s0 (- ff
                              (read-from-string s))))
    (assert (< (abs (- ff
                       (read-from-string s)))
               1d-4))
   (format nil "~af" s)))



(defun print-sufficient-digits-f64 (f)
  "print a double floating point number as a string with a given nr. of
  digits. parse it again and increase nr. of digits until the same bit
  pattern."
  (let* ((ff (coerce f 'double-float))
	 (s (format nil "~E" ff)))
    #+nil (assert (= 0d0 (- ff
			    (read-from-string s))))
    (assert (< (abs (- ff
		       (read-from-string s)))
	       1d-12))
   (substitute #\e #\d s)))
			  
(progn
  (defun emit-c (&key code (str nil)  (level 0) (hook-defun nil))
    "evaluate s-expressions in code, emit a string. if hook-defun is not nil, hook-defun will be called with every function definition. this functionality is intended to collect function declarations."
    (flet ((emit (code &optional (dl 0))
	     "change the indentation level. this is used in do"
	     (emit-c :code code :level (+ dl level) :hook-defun hook-defun)))
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
		  (paren
		   ;; paren {args}*
		   (let ((args (cdr code)))
		     (format nil "(~{~a~^, ~})" (mapcar #'emit args))))
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
		  (do0 (with-output-to-string (s)
			 ;; do0 {form}*
			 ;; write each form into a newline, keep current indentation level
			 (format s "~{~&~a~}"
				 (mapcar
				  #'(lambda (x)
				      (let ((b (emit `(indent ,x) 0)))
					(format nil "~a~a"
						b
						;; don't add semicolon if there is already one
						;; or if x contains a string
						;; or if x is an s-expression with a c thing that doesn't end with semicolon
						(if (or (eq #\; (aref b (- (length b) 1)))
							(and (typep x 'string))
							(and (listp x)
							     (member (car x) `(defun if for include dotimes while case))))
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
			     (with-output-to-string (s)
			       (loop for e in args do
				  ;; emit string if first character is not <
				    (format s "~&#include ~a"
					    (emit (if (eq #\< (aref (format nil "~a" e) 0))
						      e
						      `(string ,e))))))))
		  (progn (with-output-to-string (s)
			   ;; progn {form}*
			   ;; like do but surrounds forms with braces.
			   (format s "{~{~&~a~}~&}" (mapcar #'(lambda (x) (emit `(indent (do0 ,x)) 1)) (cdr code)))))
		  (do (with-output-to-string (s)
			;; do {form}*
			;; print each form on a new line with one more indentation.
			(format s "~{~&~a~}" (mapcar #'(lambda (x) (emit `(indent (do0 ,x)) 1)) (cdr code)))))
		  (defclass
			;; defclass class-name ({superclass-name}*) ({slot-specifier}*) [[class-option]]
			;; class TA : public Faculty, public Student { ... } 
			(destructuring-bind (name parents &rest body) (cdr code)
			  (format nil "class ~a ~@[: ~a~] ~a"
				  (emit name)
				  (when parents
				    (emit `(comma ,parents)))
				  (emit `(progn ,@body))
				  )))
		  (protected (format nil "protected ~a" (emit (cadr code))))
		  (public (format nil "public ~a" (emit (cadr code))))
		  (defun
		      (prog1
			  (parse-defun code #'emit)
			(when hook-defun
			  (funcall hook-defun (parse-defun code #'emit :header-only t)))))
		  (return (format nil "return ~a" (emit (car (cdr code)))))
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
		  (<= (destructuring-bind (a b) (cdr code)
			(format nil "(~a)<=(~a)" (emit a) (emit b))))
		  (!= (destructuring-bind (a b) (cdr code)
			(format nil "(~a)!=(~a)" (emit a) (emit b))))
		  (== (destructuring-bind (a b) (cdr code)
			(format nil "(~a)==(~a)" (emit a) (emit b))))
		  (< (destructuring-bind (a b) (cdr code)
		       (format nil "~a<~a" (emit a) (emit b))))
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
		  (char (format nil "'~a'" (cadr code)))
		  (hex (destructuring-bind (number) (cdr code)
			 (format nil "0x~x" number)))
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
						   `(do0
						     ,@(mapcar #'emit
							       forms))))
					  (format nil "~a: ~a"
						  (emit key)
						  (emit
						   `(do0
						     ,@(mapcar #'emit
							       forms))))))))))))
		  (for (destructuring-bind ((start end iter) &rest body) (cdr code)
			 (format nil "for (~@[~a~];~@[~a~];~@[~a~]) ~a"
				 (emit start)
				 (emit end)
				 (emit iter)
				 (emit `(progn ,@body)))))
		  (dotimes (destructuring-bind ((i n &optional (step 1)) &rest body) (cdr code)
			     (emit `(for (,(format nil "int ~a = 0" (emit i))
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
		  (while  ;; while condition {forms}*
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
							   (format nil "~a ~a;" (emit type) (emit slot-name))))))
					      
					      )
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
			     (format nil "~a~a" name
				     (emit `(paren ,@args)))))))))
	      (cond
		((or (symbolp code)
		     (stringp code)) ;; print variable
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
			   (format nil "{~a}" ;; remove trailing comma
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
