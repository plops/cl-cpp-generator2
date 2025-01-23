(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more ;; command line parsing
						    
						    )))
  (setf *features* (set-exclusive-or *features* (list :more
						      ;:invert
						      ))))

(let ()
  (load "util.lisp")
  (defparameter *source-dir* #P"example/163_value_based_oop/source05/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (defun begin (arr)
    `(ref (aref ,arr 0)) )
  (defun end (arr)
    `(+ (ref (aref ,arr 0)) (dot ,arr (size))))
  
  (defun lprint (&key (msg "")
		   (vars nil)
		   )
    `(<< std--cout
	 (std--format
	  (string ,(format nil "(~a~{:~a '{}'~^ ~})\\n"
			   msg
			   (loop for e in vars collect (emit-c :code e  :omit-redundant-parentheses t)) ))
	  ,@vars)))

  (let* ((class-name `Gowap)
	 (members0 `(
		     (:name frame-nr :type  int)
		     ))
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
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name class-name
     :headers `()
     :header-preamble `(do0 (comments "header")
			    (include<> 
				       string
				       vector
				       ))
     :implementation-preamble
     `(do0
       (comments "implementation")
       (include<>
	iostream
	))
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
		  (values :constructor))
		
		 
		 )

	       (defmethod ,(format nil "~~~a" class-name) ()
		 (declare
		  (values :constructor)))
	       
	       
	       ,@(remove-if
		  #'null
	          (loop for e in members
			appending
			(destructuring-bind (&key name type param doc initform param-name member-name) e
			  (let ((get (cl-change-case:pascal-case (format nil "get-~a" name)))
				(set (cl-change-case:pascal-case (format nil "set-~a" name)))
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
	       (defclass Interface ()
		 "public:"
		 (defmethod draw ()
		   (declare (virtual))))
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (&key name type param doc initform param-name member-name) e
				    `(space ,type ,member-name))))))
     :format t))
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
					;iostream
					;format
					;vector
      memory
      type_traits
      )

          
     
     (defun main ()
       (declare (values int)))

     )
   :omit-parens t
   :format t
   :tidy nil))
