
#-nil
(progn (ql:quickload "alexandria")
       (defpackage :cl-cpp-generator2
	 (:use :cl
	       :alexandria)
	 (:export
	  #:write-source)))
(in-package :cl-cpp-generator2)

(setf (readtable-case *readtable*) :invert)

(defparameter *file-hashes* (make-hash-table))

(defun write-source (name code &optional (dir (user-homedir-pathname))
				 ignore-hash)
  (let* ((fn (merge-pathnames (format nil "~a.kt" name)
			      dir))
	(code-str (emit-c
		   :code code))
	(fn-hash (sxhash fn))
	 (code-hash (sxhash code-str)))
    (multiple-value-bind (old-code-hash exists) (gethash fn-hash *file-hashes*)
     (when (or (not exists) ignore-hash (/= code-hash old-code-hash))
       ;; store the sxhash of the c source in the hash table
       ;; *file-hashes* with the key formed by the sxhash of the full
       ;; pathname
       (setf (gethash fn-hash *file-hashes*) code-hash)
       (with-open-file (s fn
			  :direction :output
			  :if-exists :supersede
			  :if-does-not-exist :create)
	 (write-sequence code-str s))
       (sb-ext:run-program "/usr/bin/clang-format"
			   (list "-i"  (namestring fn)))))))

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

(defun parse-let (code emit style)
  "let ({var | (var [init-form])}*) declaration* form*"
  ;; style can be val or var
  (destructuring-bind (decls &rest body) (cdr code)
    (multiple-value-bind (body env) (consume-declare body)
      (with-output-to-string (s)
	(format s "~a"
		(funcall emit
			`(do0
			  ,@(loop for decl in decls collect
				 (if (listp decl) ;; split into name and initform
				     (destructuring-bind (name &optional value) decl
				       (format nil "~a ~a~@[: ~a~]~@[ = ~a~]"
					       style
					       (funcall emit name)
					       (let ((type (lookup-type name :env env)))
						 (if type
						     (funcall emit type)
						     type))
					       (funcall emit value)))
				     (format nil "~a ~a: ~a"
					     style
					     decl
					     (let ((type (lookup-type decl :env env)))
					       (if type
						   (funcall emit type)
						   (break "type ~a not defined." decl))))))
			  ,@body)))))))

(defun parse-defun (code emit)
  ;;  defun function-name lambda-list [declaration*] form*
  ;; https://kotlinlang.org/docs/reference/basic-syntax.html
  ;; fun onCreate(savedInstanceState: Bundle?) { .. }
  ;; fun sum(a: Int, b: Int): Int { .. }
  
  (destructuring-bind (name lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env) (consume-declare body) ;; py
      (multiple-value-bind (req-param opt-param res-param
				      key-param other-key-p
				      aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(with-output-to-string (s)
	  (format s "fun ~a~a~@[: ~a ~]"
		  name
		  (funcall emit `(paren
				  ,@(loop for p in req-param collect
					 (format nil "~a~@[: ~a~]"
						 p
						 (let ((type (gethash p env)))
						   (if type
						       type
						       #+nil (break "can't find type for ~a in defun"
							      p)))))))
		  (let ((r (gethash 'return-values env)))
		    (if (< 1 (length r))
			(funcall emit `(paren ,@r))
			(car r))))
	  (format s "~a" (funcall emit `(progn ,@body))))))))

(defun parse-lambda (code emit)
  ;;  lambda lambda-list [declaration*] form*
  (destructuring-bind (lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env) (consume-declare body)
      (multiple-value-bind (req-param opt-param res-param
				      key-param other-key-p
				      aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(with-output-to-string (s)
	  (format s "fun ~a~@[: ~a ~]"
		  (funcall emit `(paren
				  ,@(loop for p in req-param collect
					 (format nil "~a~@[: ~a~]"
						 p
						 (let ((type (gethash p env)))
						   (if type
						       type
						       #+nil (break "can't find type for ~a in defun"
							      p)))))))
		  (let ((r (gethash 'return-values env)))
		    (if (< 1 (length r))
			(funcall emit `(paren ,@r))
			(car r))))
	  (format s "~a" (funcall emit `(progn ,@body))))))))

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
  (defun emit-c (&key code (str nil)  (level 0))
    (flet ((emit (code &optional (dl 0))
	     "change the indentation level. this is used in do"
	     (emit-c :code code :level (+ dl level))))
      (if code
	  (if (listp code)
	      (case (car code)
		(comma
		 ;; comma {args}*
		 (let ((args (cdr code)))
		   (format nil "~{~a~^, ~}" (mapcar #'emit args))))
		(paren
		 ;; paren {args}*
		 (let ((args (cdr code)))
		   (format nil "(~a)" (emit `(comma ,args)))))
		(bracket
		 ;; bracket {args}*
		 (let ((args (cdr code)))
		   (format nil "[~a]" (emit `(comma ,args)))
		   ))
		(curly
		 ;; curly {args}*
		 (let ((args (cdr code)))
		   (format nil "{~a}" (emit `(comma ,args)))))
		(indent
		 ;; indent form
		 (format nil "~{~a~}~a"
			 ;; print indentation characters
			 (loop for i below level collect "    ")
			 (emit (cadr code))))
		(do0 (with-output-to-string (s)
		       ;; do0 {form}*
		       ;; write each form into a newline, keep current indentation level
		       (format s "~&~a;~{~&~a;~}"
			       (emit (cadr code))
			       (mapcar #'(lambda (x) (emit `(indent ,x) 0)) (cddr code)))))
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
			 (format s "{~{~&~a;~}~&}" (mapcar #'(lambda (x) (emit `(indent ,x) 1)) (cdr code)))))
		(do (with-output-to-string (s)
		      ;; do {form}*
		      ;; print each form on a new line with one more indentation.
		      (format s "~{~&~a;~}" (mapcar #'(lambda (x) (emit `(indent ,x) 1)) (cdr code)))))
		(defclass
		      ;; defclass class-name ({superclass-name}*) ({slot-specifier}*) [[class-option]]
		      ;; class TA : public Faculty, public Student { ... } 
		      (destructuring-bind (name parents &rest body) (cdr code)
			(format nil "class ~a ~@[: ~a~] ~a"
				(emit name)
				(emit `(comma parents))
				(emit `(progn ,@body))
				)))
		(override (format nil "override ~a" (emit (cadr code))))
		(protected (format nil "protected ~a" (emit (cadr code))))
		(public (format nil "public ~a" (emit (cadr code))))
		(defun (parse-defun code #'emit))
		(return (format nil "return ~a" (emit (car (cdr code)))))
		(let (parse-let code #'emit 'val))
		(let-var (parse-let code #'emit 'var))
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
		(+ (let ((args (cdr code)))
		     ;; + {summands}*
		     (format nil "(~{(~a)~^+~})" (mapcar #'emit args))))
		(- (let ((args (cdr code)))
		     (if (eq 1 (length args))
			 (format nil "(-(~a))" (emit (car args))) ;; py
			 (format nil "(~{(~a)~^-~})" (mapcar #'emit args)))))
		(* (let ((args (cdr code)))
		     (format nil "(~{(~a)~^*~})" (mapcar #'emit args))))
		(/ (let ((args (cdr code)))
		     (if (eq 1 (length args))
			 (format nil "(1.0/(~a))" (emit (car args))) ;; py
			 (format nil "(~{(~a)~^/~})" (mapcar #'emit args)))))
		(logior (let ((args (cdr code))) ;; py
			  (format nil "(~{(~a)~^ or ~})" (mapcar #'emit args))))
		(logand (let ((args (cdr code))) ;; py
			  (format nil "(~{(~a)~^ and ~})" (mapcar #'emit args))))
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
		(<< (destructuring-bind (a b) (cdr code)
		      (format nil "~a<<~a" (emit a) (emit b))))
		(>> (destructuring-bind (a b) (cdr code)
		      (format nil "~a>>~a" (emit a) (emit b))))
		(incf (destructuring-bind (a &optional b) (cdr code) ;; py
			(if b
			    (format nil "(~a)+=(~a)" (emit a) (emit b))
			    (format nil "(~a)++" (emit a)))))
		(decf (destructuring-bind (a &optional b) (cdr code)
			(if b
			    (format nil "(~a)-=(~a)" (emit a) (emit b))
			    (format nil "(~a)--" (emit a)))))
		(string (format nil "\"~a\"" (cadr code)))
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
		(aref (destructuring-bind (name &rest indices) (cdr code)
			      (format nil "~a[~{~a~^,~}]" (emit name) (mapcar #'emit indices))))
		(dot (let ((args (cdr code)))
		       ;; special case: when a single ? is observed, dont print point
		       (with-output-to-string (s)
			 (loop for i below (length args) do
			      (format s "~a" (emit (elt args i)))
			      (unless (or (and
					   (< (+ i 1) (length args))
					   (or (eq (elt args (+ i 1)) '?)
					       (and (listp (elt args (+ i 1)))
						(eq (car (elt args (+ i 1))) 'progn))))
					  (eq i (1- (length args))))
				(format s "."))))
		       ;(format nil "~{~a~^.~}" (mapcar #'emit args))
		       ))
		(lambda (parse-lambda code #'emit))
		(as (let ((args (cdr code)))
		      (format nil "~a as ~a" (emit (car args)) (emit (cadr args)))))
		(case
		    ;; case keyform {normal-clause}* [otherwise-clause]
		    ;; normal-clause::= (keys form*) 
		    ;; otherwise-clause::= (t form*) 
		    
		    (destructuring-bind (keyform &rest clauses)
			(cdr code)
		      (format
		       nil "when(~a) ~a"
		       (emit keyform)
			     (emit
			      `(progn
				 ,@(loop for c in clauses collect
					(destructuring-bind (key &rest forms) c
					  (if (eq key t)
					      (format nil "else -> ~a"
						      (emit
						       `(do0
							 ,@(mapcar #'emit
								   forms))))
					      (format nil "~a -> ~a"
						      (emit key)
						      (emit
						       `(do0
							 ,@(mapcar #'emit
								   forms))))))))))))
		(for (destructuring-bind ((item collection) &rest body) (cdr code)
		       (format nil "for (~a in ~a) ~a"
			       (emit item)
			       (emit collection)
			       (emit `(progn ,@body)))))
		(t (destructuring-bind (name &rest args) code

		     (if (listp name)
			 ;; lambda call and similar complex constructs
			 (format nil "(~a)(~a)"
				 (emit name)
				 (if args
				     (emit `(paren ,@args))
				     ""))
			 ;; function call
			 
			 
			 (progn	;if
			   #+nil(and
				 (= 1 (length args))
				 (eq (aref (format nil "~a" (car args)) 0) #\.))
			   #+nil (format nil "~a~a" name
					 (emit args))


			   
			   
			   (format nil "~a~a" name
				   (emit `(paren ,@args))))))))
	      (cond
		((or (symbolp code)
		     (stringp code)) ;; print variable
		 (format nil "~a" code))
		((numberp code) ;; print constants
		 (cond ((integerp code) (format str "~a" code))
		       ((floatp code) ;; FIXME arbitrary precision?
			(format str "(~a)" (print-sufficient-digits-f64 code)))))))
	  "")))
  #+nil(progn
   (defparameter *bla*
     (emit-kt :code `(do0
		      )))
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
		      (defstruct0
			  ;; defstruct without init-form
			  ;; defstruct name {slot-description}*
			  ;; slot-description::= slot-name | (slot-name [slot-type])

			  ;; a slot-name without type can be used to create a
			  ;; composed type with a struct embedding
			  (destructuring-bind (name &rest slot-descriptions) (cdr code)
			    (format nil "type ~a struct ~a"
				    name
				    (emit
				     `(progn
					,@(loop for desc in slot-descriptions collect
					       (destructuring-bind (slot-name &optional type) desc
						 (format nil "~a~@[ ~a~]" slot-name type))))))))

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

		      (while
			  ;; while condition {forms}*
			  
			  (destructuring-bind (condition &rest body) (cdr code)
			    (with-output-to-string (s)
			      (format s "for ~a "
				      (emit condition))
			      (format s "~a" (emit `(progn ,@body))))))
		      (dotimes (destructuring-bind ((var end) &rest body) (cdr code)
				 (emit `(for ((:= ,var 0)
					      (< ,var ,end)
					      (incf ,var))
					     ,@body))))
      
		      (char (format nil "'~a'" (cadr code)))
		      (slice (let ((args (cdr code)))
			       (if (null args)
				   (format nil ":")
				   (format nil "~{~a~^:~}" (mapcar #'emit args)))))
		      
		      #+nil (-> (let ((forms (cdr code)))
				  ;; clojure's thread first macro, thrush operator
				  ;; http://blog.fogus.me/2010/09/28/thrush-in-clojure-redux/
				  ;; -> {form}*
				  (emit (reduce #'(lambda (x y) (list (emit x) (emit y))) forms)))))
