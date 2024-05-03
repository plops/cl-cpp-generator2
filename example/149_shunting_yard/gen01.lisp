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
		     (:name op :type Operator :initform nil :param nil)
		     
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
						 (unless initform
						   param-name))))
					&key
					,@(remove-if
					   #'null
					   (loop for e in members
						 collect
						 (destructuring-bind (&key name type param initform param-name member-name) e
						   (when initform
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
	   (stkOutput.push_back (Symbol (std--string 1 c)
				       Type--Literal_Numeric)))))
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))
