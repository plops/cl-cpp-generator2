(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ()
  (defparameter *source-dir* #P"example/149_shunting_yard/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  
  (let* ((class-name `Operator)
	 (members0 `((:name precedence :type uint8_t :initform 0 :param t)
		     (:name arguments :type uint8_t :initform 0 :param t)
		     ))
	 (members (loop for e in members0
			collect
			(destructuring-bind (&key name type param (initform 0)) e
			  `(:name ,name
			    :type ,type
			    :param ,param
			    :initform ,initform
			    :member-name ,(intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))
			    :param-name ,(when param
					   (intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))))))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name class-name
     :headers `()
     :header-preamble `(do0 (comments "heaeder")
			    (include <cstdint>))
     :implementation-preamble
     `(do0
       (comments "implementation"))
     :code `(do0
	     
	     (defclass ,class-name ()
	       "public:"
	       ;; handle positional arguments, followed by optional arguments
	       (defmethod ,class-name (&key ,@(remove-if
					       #'null
					       (loop for e in members
						     collect
						     (destructuring-bind (&key name type param initform param-name member-name) e
						       `(,param-name 0)))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (&key name type param initform param-name member-name) e
				       (when param
					 `(type ,type ,param-name)))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (&key name type param initform param-name member-name) e
					(cond
					  (param
					   `(,member-name ,param-name)) 
					  (initform
					   `(,member-name ,initform)))))))
		  ;(explicit)	    
		  (values :constructor))
		 )
	       ,@(remove-if
		  #'null
	          (loop for e in members
			appending
			(destructuring-bind (&key name type param initform param-name member-name) e
			  (let ((get (cl-change-case:pascal-case (format nil "get-~a" name)))
				(set (cl-change-case:pascal-case (format nil "set-~a" name)))
				(const-p (let* ((s  (format nil "~a" type))
						(last-char (aref s (- (length s)
								      1))))
					   (not (eq #\* last-char)))))
			    `((defmethod ,get ()
				(declare ,@(if const-p
					       `((const)
						 (values ,(format nil "const ~a&" type)))
					       `((values ,type))))
				(return ,member-name))
			      (defmethod ,set (,member-name)
				(declare (type ,type ,member-name))
				(setf (-> this ,member-name)
				      ,member-name)))))))
	       "public:"
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (&key name type param initform param-name member-name) e
				    `(space ,type ,member-name))))))))

  (let* ((class-name `Symbol)
	 (members0 `((:name symbol :type "std::string" :initform (string "") :param t)
		     (:name type :type "Type" :initform "Type::Unknown" :param t)
		     (:name op :type Operator :initform (curly) :param t)
		     
		     ))
	 (members (loop for e in members0
			collect
			(destructuring-bind (&key name type param (initform 0)) e
			  `(:name ,name
			    :type ,type
			    :param ,param
			    :initform ,initform
			    :member-name ,(intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))
			    :param-name ,(when param
					   (intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))))))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name class-name
     :headers `()
     :header-preamble `(do0 (comments "heaeder")
			    (include<> cstdint string)
			    (include Operator.h)
			    (space enum
		    (defclass+ Type uint8_t
		      (space-n (comma Unknown
				      Literal_Numeric
				      Operator))
		      ); type "= Type::Unknown"
		    ))
     :implementation-preamble
     `(do0
       (comments "implementation"))
     :code `(do0
	     
	     (defclass ,class-name ()
	       "public:"
	       
	       (defmethod ,class-name ( ,@(remove-if
					   #'null
					   (loop for e in members
						 collect
						 (destructuring-bind (&key name type param initform param-name member-name) e
						 (unless (and param initform)
						   param-name))))
					&key
					,@(remove-if
					   #'null
					   (loop for e in members
						 collect
						 (destructuring-bind (&key name type param initform param-name member-name) e
						   (when (and param initform)
						     `(,param-name ,initform))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (&key name type param initform param-name member-name) e
				       (when param
					 `(type ,type ,param-name)))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (&key name type param initform param-name member-name) e
					(cond
					  (param
					   `(,member-name ,param-name)) 
					  (initform
					   `(,member-name ,initform)))))))
		  ;(explicit)	    
		  (values :constructor))
		 )
	       ,@(remove-if
		  #'null
	          (loop for e in members
			appending
			(destructuring-bind (&key name type param initform param-name member-name) e
			  (let ((get (cl-change-case:pascal-case (format nil "get-~a" name)))
				(set (cl-change-case:pascal-case (format nil "set-~a" name)))
				(const-p (let* ((s  (format nil "~a" type))
						(last-char (aref s (- (length s)
								      1))))
					   (not (eq #\* last-char)))))
			    `((defmethod ,get ()
				(declare ,@(if const-p
					       `((const)
						 (values ,(format nil "const ~a&" type)))
					       `((values ,type))))
				(return ,member-name))
			      (defmethod ,set (,member-name)
				(declare (type ,type ,member-name))
				(setf (-> this ,member-name)
				      ,member-name)))))))
	       "public:"
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (&key name type param initform param-name member-name) e
				    `(space ,type ,member-name))))))))

  
  
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
      format
      iostream
      vector
      string
      unordered_map
      deque
      )
     (include Operator.h
	      Symbol.h)

     #+nil  (defun captureAndDisplay (win screen img)
	      (declare (type "Screenshot&" screen)
		       (type "const char*" win)
		       (type "cv::Mat&" img))
	      (screen img)
	      (cv--imshow win img))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       ,(lprint :msg "start")
     
       (let ((mapOps ("std::unordered_map<char,Operator>"))))

       ,(let ((l-map `((:op / :prec 4 :n 2)
		       (:op * :prec 3 :n 2)
		       (:op + :prec 2 :n 2)
		       (:op - :prec 1 :n 2)
		       ))
	      )
	  `(do0
	    ,@(loop for e in l-map
		    collect
		    (destructuring-bind (&key op prec n) e
		      `(setf (aref mapOps (char ,op))
			     #+nil (Operator ,prec ,n)
			     (curly ,prec ,n))))))

       (comments "only single digit numbers supported for now")

       (let ((sExpression (std--string (string "1+2*4-3")))))

       (let ((stkHolding (std--deque<Symbol>))))
       (let ((stkOutput (std--deque<Symbol>))))
       (for-range
	(c sExpression)
	(cond
	  ((std--isdigit c)
	   (comments "literal straight to output. they are already in order")
	   (stkOutput.push_back (curly (std--string 1 c)
				       Type--Literal_Numeric)))
	  ((mapOps.contains c)
	   (comments "symbol is operator")
	   
	   (let ((&new_op (aref mapOps c)))
	     (while (!stkHolding.empty)
		    (comments "ensure holding stack front is an operator")
		    (let ((front (stkHolding.front)))
		      (when (== Type--Operator
				front.type)
			(if (<= new_op.precedence
				  front.op.precedence)
			    (do0 (stkOutput.push_back front)
				 (stkHolding.pop_front))
			    break))))
	     (comments "push new operator on holding stack")
	     (stkHolding.push_front (curly (std--string 1 c)
					   Type--Operator
					   new_op))))
	  (t ,(lprint :msg "error"
		      :vars `(c))
	     (return 0))
	  ))

       (while (!stkHolding.empty)
	      (stkOutput.push_back (stkHolding.front))
	      (stkHolding.pop_front))
       (do0
	,(lprint :vars `(sExpression))
	(for-range (s stkOutput)
		   ,(lprint :vars `(s.symbol))))

       (let ((stkSolve (std--deque<float>))))

       
       (for-range (inst stkOutput)
		  (case inst.type
		    (Type--Unknown
		     ,(lprint :msg "error unknown symbol"))
		    (Type--Literal_Numeric
		     (comments "place literals directly on solution stack")
		     (stkSolve.push_front (std--stod inst.symbol)))
		    (Type--Operator
		     
		     (let ((mem (std--vector<double> inst.op.arguments))
			   )
		       (comments "get the number of arguments that the operator requires from the solution stack")
		       (dotimes (a inst.op.arguments)
			 (if (stkSolve.empty)
			     (do0
			      ,(lprint :msg "error solution stack is empty but operator expects operands"
				       :vars `(a inst.op.precedence)))
			     (do0
			      (comments "top of stack is at index 0")
			      (setf (aref mem a) (aref stkSolve 0))
			      (stkSolve.pop_front)))))
		     (let ((result 0s0))
		       (comments "perform operator and store result on solution stack")
		       (when (== 2 inst.op.arguments)
			 
			 ,@(loop for e in `(/ * + -)
				 collect
				 `(when (== (aref inst.symbol 0)
					    (char ,e))
				    (setf result (,e (aref mem 1)
						     (aref mem 0))))))
		       (stkSolve.push_front result)))))
       ,(lprint :msg "finished" :vars `((aref stkSolve 0)))
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))
